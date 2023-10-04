import os
import sys
import time

import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from torch import nn
from scipy.special import gamma
from torch.autograd import Variable

sys.path.append("..")
from hermite import hermite_getAll, D_u_0, D_u_1, D_u_2


def set_seed(seed):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.GELU()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a


class Model:
    def __init__(self, order, net, x0, u0, t, lb, ub,
                 x_test, x_test_exact
                 ):

        self.H_coefficient = None
        self.x_b2 = None
        self.x_b1 = None
        self.x_b2_u = None
        self.x_b1_u = None

        self.x_f = None

        self.optimizer_LBGFS = None

        self.order = order
        self.net = net

        self.x0 = x0
        self.u0 = u0
        self.x_N = len(x0)

        self.t = t
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))
        self.lb = lb
        self.ub = ub

        self.x_test = x_test
        self.x_test_exact = x_test_exact

        self.x_b_loss_collect = []
        self.x_f_loss_collect = []

        self.x_test_estimate_collect = []

        self.init_data()

    def init_data(self):
        self.H_coefficient = hermite_getAll(self.t, order=self.order)

        temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[0])
        self.x_f = torch.cat((temp_t, self.x0), dim=1)
        for i in range(self.t_N - 1):
            temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[i + 1])
            x_f_temp = torch.cat((temp_t, self.x0), dim=1)
            self.x_f = torch.cat((self.x_f, x_f_temp), dim=0)

        temp_t = torch.from_numpy(self.t[:, None])
        temp_lb = torch.full_like(torch.zeros(self.t_N, 1), self.lb[1])
        temp_ub = torch.full_like(torch.zeros(self.t_N, 1), self.ub[1])
        self.x_b1 = torch.cat((temp_t, temp_lb), dim=1)
        self.x_b2 = torch.cat((temp_t, temp_ub), dim=1)
        self.x_b1_u = exact_u(self.x_b1)
        self.x_b2_u = exact_u(self.x_b2)

        self.lb = torch.from_numpy(self.lb)
        self.ub = torch.from_numpy(self.ub)

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H) * x[:, [0]] + x[:, [1]] ** 2

    def predict_U(self, x):
        return self.train_U(x)

    def hat_ut_and_Lu(self):
        x = Variable(self.x_f, requires_grad=True)

        u = self.train_U(x)
        d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        u_t = d[0][:, [0]]
        u_x = d[0][:, [1]]
        dd = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)
        u_xx = dd[0][:, [1]]
        u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, [0]]

        u = u.reshape(self.t_N, -1)
        u_t = u_t.reshape(self.t_N, -1)
        u_tt = u_tt.reshape(self.t_N, -1)
        Lu = u_xx.reshape(self.t_N, -1)

        return u, u_t, Lu, u_tt

    def PDE_loss(self):
        u, u_t, Lu, u_tt = self.hat_ut_and_Lu()
        if self.order == 0:
            ut = D_u_0(self.H_coefficient, self.t, u, alpha)
        if self.order == 1:
            ut = D_u_1(self.H_coefficient, self.t, u, u_t, alpha)
        if self.order == 2:
            ut = D_u_2(self.H_coefficient, self.t, u, u_t, u_tt, alpha)
        loss = torch.mean((ut[1:] - Lu[1:]) ** 2)
        return loss

    def calculate_loss(self):

        loss_b1 = torch.mean((self.train_U(self.x_b1) - self.x_b1_u) ** 2)
        loss_b2 = torch.mean((self.train_U(self.x_b2) - self.x_b2_u) ** 2)
        loss_b = loss_b1 + loss_b2
        self.x_b_loss_collect.append([self.net.iter, loss_b.item()])

        loss_f = self.PDE_loss()
        self.x_f_loss_collect.append([self.net.iter, loss_f.item()])

        return loss_b, loss_f

    # computer backward loss
    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_b, loss_f = self.calculate_loss()
        loss = loss_b + loss_f
        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item())
        return loss

    def train(self, epochs=10000):
        self.optimizer_LBGFS = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=epochs,
            max_eval=epochs,
            history_size=100,
            tolerance_grad=1e-9,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        start_time = time.time()
        self.optimizer_LBGFS.step(self.LBGFS_loss)
        print('LBGFS done!')

        pred = self.train_U(x_test).cpu().detach().numpy()
        exact = self.x_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)
        return error, elapsed, self.LBGFS_loss().item()


def draw_exact():
    predict_np = model.predict_U(x_test).cpu().detach().numpy()
    u_test_np = x_test_exact.cpu().detach().numpy()
    TT, XX = np.meshgrid(t_pred, x_pred)
    e = np.reshape(u_test_np, (TT.shape[0], TT.shape[1]))
    fig = plt.figure(1, figsize=(10, 10))
    fig.add_subplot(2, 1, 1)
    plt.pcolor(TT, XX, e, cmap='jet', shading='auto')
    plt.colorbar()
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('$x$', fontsize=20)
    plt.title(r'Exact $u(x,t)$', fontsize=20)
    plt.tight_layout()
    fig.add_subplot(2, 1, 2)
    e1 = np.reshape(predict_np, (TT.shape[0], TT.shape[1]))
    plt.pcolor(TT, XX, e1, cmap='jet', shading='auto')
    plt.colorbar()
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('$x$', fontsize=20)
    plt.title(r'Pred $u(x,t)$', fontsize=20)
    plt.tight_layout()
    plt.show()


def draw_error():
    predict_np = model.predict_U(x_test).cpu().detach().numpy()
    u_test = x_test_exact.cpu().detach().numpy()
    TT, XX = np.meshgrid(t_pred, x_pred)
    e = np.reshape(abs(predict_np - u_test), (TT.shape[0], TT.shape[1]))
    plt.pcolor(TT, XX, e, cmap='jet', shading='auto')
    plt.colorbar()
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('$x$', fontsize=20)
    plt.title('$Error$', fontsize=20)
    plt.tight_layout()
    plt.show()


def draw_epoch_loss():
    x_b_loss_collect = np.array(model.x_b_loss_collect)
    x_f_loss_collect = np.array(model.x_f_loss_collect)
    plt.yscale('log')
    plt.xlabel('$Epoch$', fontsize=20)
    plt.ylabel('$Loss$', fontsize=20)
    plt.plot(x_b_loss_collect[:, 0], x_b_loss_collect[:, 1], 'g-', label='x_b_loss')

    plt.plot(x_f_loss_collect[:, 0], x_f_loss_collect[:, 1], 'r-', label='PDE_loss')
    plt.legend()
    plt.show()


def draw_some_t():
    predict_np = model.predict_U(x_test).cpu().detach().numpy()
    u_test_np = x_test_exact.cpu().detach().numpy()
    TT, XX = np.meshgrid(t_pred, x_pred)
    u_pred = np.reshape(predict_np, (TT.shape[0], TT.shape[1]))
    u_test = np.reshape(u_test_np, (TT.shape[0], TT.shape[1]))
    gs1 = gridspec.GridSpec(2, 2)

    plot_t = int(t_pred_N / 4)
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x_pred, u_test.T[0, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x_pred, u_pred.T[0, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = %.2f$' % (t_pred[0]), fontsize=10)
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x_pred, u_test.T[plot_t, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x_pred, u_pred.T[plot_t, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = %.2f$' % (t_pred[plot_t]), fontsize=10)

    ax = plt.subplot(gs1[1, 0])
    ax.plot(x_pred, u_test.T[2 * plot_t, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x_pred, u_pred.T[2 * plot_t, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = %.2f$' % (t_pred[2 * plot_t]), fontsize=10)

    ax = plt.subplot(gs1[1, 1])
    ax.plot(x_pred, u_test.T[3 * plot_t, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x_pred, u_pred.T[3 * plot_t, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = %.2f$' % (t_pred[3 * plot_t]), fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    set_seed(1234)
    # train
    order = 1
    layers = [2, 20, 20, 20, 20, 1]
    net = Net(layers)

    alpha = 0.5
    exact_u = lambda x: x[:, [1]] ** 2 + (2 * x[:, [0]] ** alpha) / gamma(alpha + 1)

    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    # test
    t_pred_N = 100
    x_pred_N = 100
    t_pred = np.linspace(lb[0], ub[0], t_pred_N)[:, None]
    x_pred = np.linspace(lb[1], ub[1], x_pred_N)[:, None]
    t_star, x_star = np.meshgrid(t_pred, x_pred)
    t_star = t_star.flatten()[:, None]
    x_star = x_star.flatten()[:, None]
    tx = np.hstack((t_star, x_star))

    x_test = torch.from_numpy(tx)
    x_test_exact = exact_u(x_test)

    t_N = 51
    x_N = 11
    train_t = np.linspace(lb[0], ub[0], t_N)
    x0 = np.linspace(lb[1], ub[1], x_N)[:, None]
    x0 = torch.from_numpy(x0).float()
    u0 = x0 ** 2

    model = Model(
        order=order,
        net=net,
        x0=x0,
        u0=u0,
        t=train_t,
        lb=lb,
        ub=ub,
        x_test=x_test,
        x_test_exact=x_test_exact,
    )

    model.train(epochs=3000)
    draw_exact()
    draw_error()
    draw_epoch_loss()
    # draw_some_t()
