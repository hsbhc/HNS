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
                 x_test, x_test_exact, exact_u
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
        self.exact_u = exact_u
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
        self.x_b1_u = self.exact_u(self.x_b1)
        self.x_b2_u = self.exact_u(self.x_b2)

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

        pred = self.train_U(self.x_test).cpu().detach().numpy()
        exact = self.x_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)
        return error, elapsed, self.LBGFS_loss().item()


def show_single():
    global alpha
    alpha = 0.5
    order = 1

    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    t_N = 51
    x_N = 11
    train_t = np.linspace(lb[0], ub[0], t_N)
    x0 = np.linspace(lb[1], ub[1], x_N)[:, None]
    x0 = torch.from_numpy(x0).float()
    u0 = x0 ** 2

    layers = [2, 20, 20, 20, 20, 1]
    exact_u = lambda x: x[:, [1]] ** 2 + (2 * x[:, [0]] ** alpha) / gamma(alpha + 1)

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

    set_seed(1234)
    net = Net(layers)
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
        exact_u=exact_u
    )

    model.train(epochs=3000)

    predict_np = model.predict_U(model.x_test).cpu().detach().numpy()
    u_test_np = model.x_test_exact.cpu().detach().numpy()
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
    plt.title(r'Pred $\tilde{u}(x,t)$', fontsize=20)
    plt.tight_layout()
    plt.savefig('TFDE_single.pdf')
    plt.show()


def show_loss():
    global alpha
    alpha = 0.65

    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    t_N = 21
    x_N = 11
    train_t = np.linspace(lb[0], ub[0], t_N)
    x0 = np.linspace(lb[1], ub[1], x_N)[:, None]
    x0 = torch.from_numpy(x0).float()
    u0 = x0 ** 2

    layers = [2, 20, 20, 20, 20, 1]
    exact_u = lambda x: x[:, [1]] ** 2 + (2 * x[:, [0]] ** alpha) / gamma(alpha + 1)

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

    orders = [0, 1, 2]
    l2errors = []
    times = []
    x_b_loss_collects = []
    x_f_loss_collects = []
    for order in orders:
        set_seed(1234)
        net = Net(layers)
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
            exact_u=exact_u
        )
        error, elapsed, loss = model.train(epochs=3000)
        l2errors.append(error)
        times.append(elapsed)
        x_b_loss_collect = np.array(model.x_b_loss_collect)
        x_f_loss_collect = np.array(model.x_f_loss_collect)
        x_b_loss_collects.append(x_b_loss_collect)
        x_f_loss_collects.append(x_f_loss_collect)

    plt.rc('legend', fontsize=16)
    fig = plt.figure(figsize=(10, 4))
    fig.add_subplot(1, 3, 1)
    plt.yscale('log')
    plt.xlabel('$Epoch$', fontsize=20)
    plt.ylabel('$Loss$', fontsize=20)
    plt.plot(x_b_loss_collects[0][:, 0], x_b_loss_collects[0][:, 1] + x_f_loss_collects[0][:, 1], 'b-', linewidth=2,
             label='p=1')
    plt.plot(x_b_loss_collects[1][:, 0], x_b_loss_collects[1][:, 1] + x_f_loss_collects[1][:, 1], 'g-', linewidth=2,
             label='p=3')
    plt.plot(x_b_loss_collects[2][:, 0], x_b_loss_collects[2][:, 1] + x_f_loss_collects[2][:, 1], 'y-', linewidth=2,
             label='p=5')
    plt.legend()
    plt.tight_layout()

    fig.add_subplot(1, 3, 2)
    labels = ["1", "3", "5"]
    plt.bar(labels, times, fc="#96f97b")
    plt.xlabel("p", fontsize=20)
    plt.ylabel("Time", fontsize=20)
    plt.tight_layout()

    fig.add_subplot(1, 3, 3)
    labels = ["1", "3", "5"]
    plt.bar(labels, l2errors, fc="#ff81c0")
    plt.xlabel("p", fontsize=20)
    plt.ylabel("$L2$error", fontsize=20)
    plt.tight_layout()

    plt.savefig('TFDE_loss.pdf')
    plt.show()


def show_NxNt():
    global alpha
    alpha = 0.65
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    layers = [2, 20, 20, 20, 20, 1]
    exact_u = lambda x: x[:, [1]] ** 2 + (2 * x[:, [0]] ** alpha) / gamma(alpha + 1)

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

    x_N = 11
    x0 = np.linspace(lb[1], ub[1], x_N)[:, None]
    x0 = torch.from_numpy(x0).float()
    u0 = x0 ** 2

    orders = [0, 1, 2]
    t_N_all = np.array([6, 11, 21, 41, 81, 101])

    error_all = np.zeros([len(t_N_all), len(orders)])

    for i in range(len(t_N_all)):
        for j in range(len(orders)):
            t_N = t_N_all[i]
            order = orders[j]
            train_t = np.linspace(lb[0], ub[0], t_N)

            set_seed(1234)
            net = Net(layers)
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
                exact_u=exact_u
            )
            error, elapsed, loss = model.train(epochs=3000)
            error_all[i, j] = error

    t_N = 41
    train_t = np.linspace(lb[0], ub[0], t_N)

    orders = [0, 1, 2]
    x_N_all = np.array([6, 11, 21, 31, 41, 51])

    error_all1 = np.zeros([len(x_N_all), len(orders)])

    for i in range(len(x_N_all)):
        for j in range(len(orders)):
            x_N = x_N_all[i]
            order = orders[j]

            x0 = np.linspace(lb[1], ub[1], x_N)[:, None]
            x0 = torch.from_numpy(x0).float()
            u0 = x0 ** 2

            set_seed(1234)
            net = Net(layers)
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
                exact_u=exact_u
            )
            error, elapsed, loss = model.train(epochs=3000)
            error_all1[i, j] = error

    plt.rc('legend', fontsize=16)

    fig = plt.figure(figsize=(10, 4))
    fig.add_subplot(1, 2, 1)
    plt.yscale('log')
    plt.xlabel('$M_t$', fontsize=20)
    plt.title('relative $L2$ error, $M_x = 11$', fontsize=20)
    plt.plot(t_N_all, error_all[:, 0], 'b-', marker="*", linewidth=2, label='p=1')
    plt.plot(t_N_all, error_all[:, 1], 'g-', marker="*", linewidth=2, label='p=3')
    plt.plot(t_N_all, error_all[:, 2], 'y-', marker="*", linewidth=2, label='p=5')
    plt.legend()
    plt.tight_layout()

    fig.add_subplot(1, 2, 2)
    plt.yscale('log')
    plt.xlabel('$M_x$', fontsize=20)
    plt.title('relative $L2$ error, $M_t = 41$', fontsize=20)
    plt.plot(x_N_all, error_all1[:, 0], 'b-', marker="*", linewidth=2, label='p=1')
    plt.plot(x_N_all, error_all1[:, 1], 'g-', marker="*", linewidth=2, label='p=3')
    plt.plot(x_N_all, error_all1[:, 2], 'y-', marker="*", linewidth=2, label='p=5')
    plt.legend()
    plt.tight_layout()

    plt.savefig('TFDE_NxNt.pdf')
    plt.show()


def show_all():
    global alpha
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    layers = [2, 20, 20, 20, 20, 1]

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

    x_N = 11
    x0 = np.linspace(lb[1], ub[1], x_N)[:, None]
    x0 = torch.from_numpy(x0).float()
    u0 = x0 ** 2

    alphas = [0.45, 0.65, 0.85]
    orders = [0, 1, 2]
    t_N_all = np.array([6, 11, 21, 41, 61, 81, 101])

    error_all = np.zeros([len(t_N_all), len(orders) * len(alphas)])

    for i in range(len(alphas)):
        alpha = alphas[i]
        exact_u = lambda x: x[:, [1]] ** 2 + (2 * x[:, [0]] ** alpha) / gamma(alpha + 1)
        x_test_exact = exact_u(x_test)

        for j in range(len(t_N_all)):
            for k in range(len(orders)):
                t_N = t_N_all[j]
                order = orders[k]

                train_t = np.linspace(lb[0], ub[0], t_N)

                set_seed(1234)
                net = Net(layers)
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
                    exact_u=exact_u
                )
                error, elapsed, loss = model.train(epochs=3000)
                error_all[j, i * len(orders) + k] = error

    # np.savetxt('TFDE_all.txt',error_all)

    file = open('TFDE_all.txt', 'w+')
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
        file.write(
            '\\cmidrule(r){' + str(i * len(orders) + 2) + '-' + str(i * len(orders) + 1 + len(orders)) + '}' + ' ')
    file.write('\n' + '$M_t$')

    for i in range(len(alphas)):
        for j in range(len(orders)):
            file.write('& $p=' + str(2 * orders[j] + 1) + '$' + ' ')
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

    # show_single()

    show_loss()

    # show_NxNt()

    # show_all()
