import os
import sys
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
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
    def __init__(self, order, net, u0, t, lb, ub,
                 x_test, x_test_exact
                 ):

        self.H_coefficient = None
        self.x_f = None
        self.optimizer_LBGFS = None

        self.order = order
        self.net = net

        self.u0 = u0

        self.t = t
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))
        self.lb = lb
        self.ub = ub

        self.x_test = x_test
        self.x_test_exact = x_test_exact

        self.x_f_loss_collect = []

        self.x_test_estimate_collect = []

        self.init_data()

    def init_data(self):
        self.H_coefficient = hermite_getAll(self.t, order=self.order)

        self.x_f = torch.from_numpy(self.t).unsqueeze(-1)

        self.lb = torch.from_numpy(self.lb).float()
        self.ub = torch.from_numpy(self.ub).float()

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H) * x + 1

    def predict_U(self, x):
        return self.train_U(x)

    def hat_ut_and_Lu(self):
        x = Variable(self.x_f, requires_grad=True)

        u = self.train_U(x)
        d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        u_t = d[0][:, [0]]
        u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, [0]]

        Lu = u + gamma(3) / gamma(3 - alpha) * self.x_f ** (2 - alpha) - self.x_f ** 2 - 1
        u = u.reshape(self.t_N, -1)
        u_t = u_t.reshape(self.t_N, -1)
        u_tt = u_tt.reshape(self.t_N, -1)
        Lu = Lu.reshape(self.t_N, -1)

        return u, u_t, Lu, u_tt

    def PDE_loss(self):
        u, u_t, Lu, u_tt = self.hat_ut_and_Lu()
        if self.order == 0:
            ut = D_u_0(self.H_coefficient, self.t, u,alpha)
        if self.order == 1:
            ut = D_u_1(self.H_coefficient, self.t, u, u_t,alpha)
        if self.order == 2:
            ut = D_u_2(self.H_coefficient, self.t, u, u_t, u_tt,alpha)

        loss = torch.mean((ut[1:] - Lu[1:]) ** 2)
        return loss

    def calculate_loss(self):
        loss_f = self.PDE_loss()
        self.x_f_loss_collect.append([self.net.iter, loss_f.item()])
        return loss_f

    # computer backward loss
    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss = self.calculate_loss()
        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item())
        return loss

    def train(self, epochs=50000):
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


def show_single():
    global alpha

    alpha = 0.5
    t_N = 3
    lb = np.array([0.0])
    ub = np.array([2.0])

    # test
    t_pred_N = 1000
    t_pred = np.linspace(lb[0], ub[0], t_pred_N)[:, None]
    x_test = torch.from_numpy(t_pred)
    x_test_exact = exact_u(x_test)

    train_t = np.linspace(lb[0], ub[0], t_N)
    u0 = exact_u(x_test[0])

    predict = []
    for order in [0, 1, 2]:
        set_seed(1234)
        net = Net(layers)
        model = Model(
            order=order,
            net=net,
            u0=u0,
            t=train_t,
            lb=lb,
            ub=ub,
            x_test=x_test,
            x_test_exact=x_test_exact,
        )
        model.train(epochs=1000)
        predict.append(model.predict_U(x_test).cpu().detach().numpy())

    u_test_np = x_test_exact.cpu().detach().numpy()
    plt.rc('legend', fontsize=16)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_pred, u_test_np, 'b-', linewidth=1, label='Exact')
    plt.plot(t_pred, predict[0], 'g--', linewidth=1, label='p=1')
    plt.plot(t_pred, predict[1], 'y--', linewidth=1, label='p=3')
    plt.plot(t_pred, predict[2], 'r--', linewidth=1, label='p=5')
    plt.xlabel('$t$', fontsize=20)
    plt.title(r'$u(t)$', fontsize=20)
    plt.legend()
    plt.tight_layout()

    alpha = 0.5
    t_N = 6
    lb = np.array([0.0])
    ub = np.array([2.0])

    # test
    t_pred_N = 1000
    t_pred = np.linspace(lb[0], ub[0], t_pred_N)[:, None]
    x_test = torch.from_numpy(t_pred)
    x_test_exact = exact_u(x_test)

    train_t = np.linspace(lb[0], ub[0], t_N)
    u0 = exact_u(x_test[0])

    predict1 = []
    for order in [0, 1, 2]:
        set_seed(1234)
        net = Net(layers)
        model = Model(
            order=order,
            net=net,
            u0=u0,
            t=train_t,
            lb=lb,
            ub=ub,
            x_test=x_test,
            x_test_exact=x_test_exact,
        )
        model.train(epochs=1000)
        predict1.append(model.predict_U(x_test).cpu().detach().numpy())

    u_test_np = x_test_exact.cpu().detach().numpy()
    plt.subplot(1, 2, 2)
    plt.plot(t_pred, u_test_np, 'b-', linewidth=1, label='Exact')
    plt.plot(t_pred, predict1[0], 'g--', linewidth=1, label='p=1')
    plt.plot(t_pred, predict1[1], 'y--', linewidth=1, label='p=3')
    plt.plot(t_pred, predict1[2], 'r--', linewidth=1, label='p=5')
    plt.xlabel('$t$', fontsize=20)
    plt.title(r'$u(t)$', fontsize=20)
    plt.legend()
    plt.tight_layout()

    plt.savefig('FDE_single.pdf')
    plt.show()


def show_all():
    global alpha

    alphas = [0.3, 0.5, 0.7]
    order = [0, 1, 2]
    t_N_all = np.array([6, 11, 21, 41, 81, 101])

    error_all = np.zeros([len(t_N_all), len(order) * len(alphas)])

    for i in range(len(alphas)):
        for j in range(len(t_N_all)):
            for k in range(len(order)):
                alpha = alphas[i]
                t_N = t_N_all[j]
                m = order[k]

                train_t = np.linspace(lb[0], ub[0], t_N)
                u0 = exact_u(x_test[0])
                set_seed(1234)
                net = Net(layers)
                model = Model(
                    order=m,
                    net=net,
                    u0=u0,
                    t=train_t,
                    lb=lb,
                    ub=ub,
                    x_test=x_test,
                    x_test_exact=x_test_exact,
                )
                error, elapsed, loss = model.train(epochs=3000)
                error_all[j, i * len(order) + k] = error

    # np.savetxt('FDE_all.txt',error_all)

    file = open('FDE_all.txt', 'w+')
    file.write('\\begin{table}' + '\n')
    file.write('\\caption{Sample table title}' + '\n')
    file.write('\\label{sample-table}' + '\n')
    file.write('\\centering' + '\n')
    file.write('\\begin{tabular}{' + 'l' * (len(error_all[0]) + 1) + '}' + '\n')
    file.write('\\toprule' + '\n')
    for i in alphas:
        file.write('& \\multicolumn{3}{c}{$\\alpha = ' + str(i) + '$}' + ' ')
    file.write('\\\\' + '\n')
    for i in range(len(alphas)):
        file.write('\\cmidrule(r){' + str(i * len(order) + 2) + '-' + str(i * len(order) + 1 + len(order)) + '}' + ' ')
    file.write('\n' + '$M_t$')

    for i in range(len(alphas)):
        for j in range(len(order)):
            file.write('& $p=' + str(2 * order[j] + 1) + '$' + ' ')
    file.write('\\\\' + '\n')
    file.write('\\midrule' + '\n')

    for i in range(error_all.shape[0]):
        file.write(str(t_N_all[i]) + ' ')
        for j in range(error_all.shape[1]):
            temp = '%.2e' % (error_all[i, j])
            file.write(' & ' + temp)
        file.write('\\\\' + '\n')
    file.write('\\bottomrule' + '\n')
    file.write('\\end{tabular}' + '\n')
    file.write('\\end{table}' + '\n')
    file.close()


if __name__ == '__main__':
    set_seed(1234)

    # train
    layers = [1, 20, 20, 20, 20, 1]
    exact_u = lambda x: 1 + x ** 2

    lb = np.array([0.0])
    ub = np.array([2.0])

    # test
    t_pred_N = 1000
    t_pred = np.linspace(lb[0], ub[0], t_pred_N)[:, None]
    x_test = torch.from_numpy(t_pred)
    x_test_exact = exact_u(x_test)

    show_single()
    # show_all()
