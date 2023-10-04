import os
import sys
import time

import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from scipy.special import gamma as gamma1
from torch.autograd import Variable
from pyDOE import lhs

sys.path.append("../..")
from hermite_inv import hermite_getAll, D_u_0, D_u_1, D_u_2


def gamma(x):
    return torch.exp(torch.lgamma(x))


def set_seed(seed):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def random_fun(Dim, lb, ub, num):
    temp = torch.from_numpy(lb + (ub - lb) * lhs(Dim, num)).float()
    return temp


class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.alpha = nn.Parameter(torch.tensor(0.2).float(), requires_grad=True)
        self.a1 = nn.Parameter(torch.tensor(0.2).float(), requires_grad=True)
        self.a2 = nn.Parameter(torch.tensor(0.2).float(), requires_grad=True)
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
    def __init__(self, order, net, xyz0, u0, t, lb, ub,
                 x_test, x_test_exact, exact_u
                 ):

        self.x_f_u = None
        self.H_coefficient = None
        self.x_b = None
        self.x_b_u = None

        self.x_f = None

        self.optimizer_LBGFS = None

        self.order = order
        self.net = net

        self.xyz0 = xyz0
        self.u0 = u0
        self.x_N = len(xyz0)

        self.t = t
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))
        self.lb = lb
        self.ub = ub

        self.x_test = x_test
        self.x_test_exact = x_test_exact
        self.exact_u = exact_u

        self.x_b_loss_collect = []
        self.x_f_loss_collect = []

        self.x_test_estimate_collect = []

        self.init_data()

    def make_boundary(self):

        temp_l = random_fun(Dim + 1, lb, ub, x_b_N)
        temp_u = random_fun(Dim + 1, lb, ub, x_b_N)
        temp_l[:, 1] = lb[1]
        temp_u[:, 1] = ub[1]
        self.x_b = torch.cat((temp_l, temp_u), dim=0)
        for i in range(2, Dim + 1):
            temp_l = random_fun(Dim + 1, lb, ub, x_b_N)
            temp_u = random_fun(Dim + 1, lb, ub, x_b_N)
            temp_l[:, i] = lb[i]
            temp_u[:, i] = ub[i]
            self.x_b = torch.cat((self.x_b, temp_l, temp_u), dim=0)

        self.x_b_u = self.exact_u(self.x_b)
        return self.x_b_u

    def init_data(self):
        self.H_coefficient = hermite_getAll(self.t, order=self.order)

        temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[0])
        self.x_f = torch.cat((temp_t, self.xyz0), dim=1)
        for i in range(self.t_N - 1):
            temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[i + 1])
            x_f_temp = torch.cat((temp_t, self.xyz0), dim=1)
            self.x_f = torch.cat((self.x_f, x_f_temp), dim=0)

        self.x_f_u = self.exact_u(self.x_f)
        self.make_boundary()
        self.lb = torch.from_numpy(self.lb)
        self.ub = torch.from_numpy(self.ub)
        # print(self.x_f)
        # print(self.x_b)
        # exit(0)

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H) * x[:, [0]]

    def predict_U(self, x):
        return self.train_U(x)

    def hat_ut_and_Lu(self):
        x = Variable(self.x_f, requires_grad=True)
        u = self.train_U(x)
        d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        u_1 = d[0]
        u_2 = []
        for i in range(Dim + 1):
            u_2.append(
                torch.autograd.grad(u_1[:, [i]], x, grad_outputs=torch.ones_like(u_1[:, [i]]), create_graph=True)[0][:,
                [i]])
        u_2 = torch.cat(u_2, dim=1)
        lu1 = net.a1 * torch.sum(u_2[:, 1:], dim=1, keepdim=True) - net.a2 * torch.sum(u_1[:, 1:], dim=1, keepdim=True)
        lu2 = torch.sum(torch.cos(x[:, 1:Dim + 1]),
                        dim=1, keepdim=True) * gamma1(3) / gamma1(3 - alpha) * x[:, [0]] ** (2 - alpha) + x[:,
                                                                                                          [
                                                                                                              0]] ** 2 * (
                      torch.sum(torch.cos(x[:, 1:Dim + 1]), dim=1,
                                keepdim=True) - torch.sum(
                  torch.sin(x[:, 1:Dim + 1]), dim=1, keepdim=True))
        Lu = lu1 + lu2
        u = u.reshape(self.t_N, -1)
        u_t = u_1[:, [0]].reshape(self.t_N, -1)
        u_tt = u_2[:, [0]].reshape(self.t_N, -1)
        Lu = Lu.reshape(self.t_N, -1)

        return u, u_t, Lu, u_tt

    def PDE_loss(self):
        u, u_t, Lu, u_tt = self.hat_ut_and_Lu()

        if self.order == 0:
            ut = D_u_0(self.H_coefficient, self.t, u, net.alpha)
        if self.order == 1:
            ut = D_u_1(self.H_coefficient, self.t, u, u_t, net.alpha)
        if self.order == 2:
            ut = D_u_2(self.H_coefficient, self.t, u, u_t, u_tt, net.alpha)
        loss = torch.mean((ut[1:] - Lu[1:]) ** 2)
        return loss

    def calculate_loss(self):

        loss_b = torch.mean((self.train_U(self.x_b) - self.x_b_u) ** 2)
        # self.x_b_loss_collect.append([self.net.iter, loss_b.item()])

        loss_f_u = torch.mean((self.train_U(self.x_f) - self.x_f_u) ** 2)
        self.x_b_loss_collect.append([self.net.iter, (loss_b + loss_f_u).item()])

        loss_f = self.PDE_loss()
        self.x_f_loss_collect.append([self.net.iter, loss_f.item()])

        return loss_b + loss_f_u, loss_f

    # computer backward loss
    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_b, loss_f = self.calculate_loss()
        loss = loss_b + loss_f
        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item(), net.alpha.item(), net.a1.item(), net.a2.item())
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

        pred = self.train_U(self.x_test).cpu().detach().numpy()
        exact = self.x_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)
        return error, elapsed, self.LBGFS_loss().item()


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


def draw():
    xy_pred_N = 100
    x_pred = np.linspace(lb[1], ub[1], xy_pred_N)[:, None]
    y_pred = np.linspace(lb[2], ub[2], xy_pred_N)[:, None]
    x_star, y_star = np.meshgrid(x_pred, y_pred)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    t = np.ones_like(x_star) * 1
    other_x = np.ones_like(x_star) * 0.5
    xy = np.hstack((t, x_star, y_star))
    for i in range(Dim - 2):
        xy = np.hstack((xy, other_x))
    x_test = torch.from_numpy(xy)

    pred = model.predict_U(x_test).detach().numpy()
    pred = pred.reshape((xy_pred_N, xy_pred_N))
    Exact = exact_u(x_test).detach().numpy()
    Exact = Exact.reshape((xy_pred_N, xy_pred_N))

    nn = 200
    x_plot = np.linspace(lb[1], ub[1], nn)
    y_plot = np.linspace(lb[2], ub[2], nn)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    xy = np.hstack((x_star, y_star))
    u_data_pred = griddata(xy, pred[:, :].reshape((-1, 1))[:, -1], (X_plot, Y_plot),
                           method='cubic')
    u_data_plot = griddata(xy, Exact[:, :].reshape((-1, 1))[:, -1], (X_plot, Y_plot),
                           method='cubic')

    fig = plt.figure()
    ax = fig.add_subplot(131)
    h = ax.imshow(u_data_plot, interpolation='nearest', vmin=(Dim - 2) * np.cos(0.5) + 2 * np.cos(1.) - 0.1,
                  vmax=(Dim - 2) * np.cos(0.5) + 2.1, cmap='seismic',
                  extent=[lb[0], ub[1], lb[2], ub[2]],
                  origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Exact $u$', fontsize=10)

    ax = fig.add_subplot(132)
    h = ax.imshow(u_data_pred, interpolation='nearest', vmin=(Dim - 2) * np.cos(0.5) + 2 * np.cos(1.) - 0.1,
                  vmax=(Dim - 2) * np.cos(0.5) + 2.1, cmap='seismic',
                  extent=[lb[0], ub[1], lb[2], ub[2]],
                  origin='lower')
    divider = make_axes_locatable(ax)
    ax.axis('off')
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    ax.set_title('Pred $\\tilde{u}$', fontsize=10)

    ax = fig.add_subplot(133)
    h = ax.imshow(abs(u_data_pred - u_data_plot), interpolation='nearest', cmap='jet',
                  extent=[lb[0], ub[1], lb[2], ub[2]],
                  origin='lower')
    divider = make_axes_locatable(ax)
    ax.axis('off')
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    ax.set_title('|$\\tilde{u}$ - $u$|', fontsize=10)

    fig.tight_layout(h_pad=1)

    plt.show()


if __name__ == '__main__':
    set_seed(1234)
    Dim = 3
    # train
    order = 0
    layers = [Dim + 1, 50, 50, 50, 50, 1]
    net = Net(layers)

    alpha = 0.5
    exact_u = lambda x: x[:, [0]] ** 2 * torch.sum(torch.cos(x[:, 1:Dim + 1]), dim=1, keepdim=True)
    exact_u0 = lambda x: torch.sum(torch.cos(x), dim=1, keepdim=True) * 0.0

    lb = np.array([0.0 for i in range(Dim + 1)])
    ub = np.array([1.0 for i in range(Dim + 1)])

    # test
    pred_N = 5000000
    x_test = random_fun(Dim + 1, lb, ub, pred_N)
    x_test_exact = exact_u(x_test)

    t_N = 6
    x_N = 1000
    x_b_N = 50
    train_t = np.linspace(lb[0], ub[0], t_N)

    xyz0 = random_fun(Dim, lb[1:], ub[1:], x_N)
    u0 = exact_u0(xyz0)

    model = Model(
        order=order,
        net=net,
        xyz0=xyz0,
        u0=u0,
        t=train_t,
        lb=lb,
        ub=ub,
        x_test=x_test,
        x_test_exact=x_test_exact,
        exact_u=exact_u
    )
    # draw()
    model.train(epochs=1000)
    #
    # draw_epoch_loss()
    # draw()
