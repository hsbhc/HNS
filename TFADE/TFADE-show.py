import os
import sys
import time
import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    def __init__(self, order, net, xy0, u0, t, lb, ub,
                 x_test, x_test_exact, exact_u
                 ):

        self.H_coefficient = None
        self.x_b = None
        self.x_b_u = None

        self.x_f = None

        self.optimizer_LBGFS = None

        self.order = order
        self.net = net

        self.xy0 = xy0
        self.u0 = u0
        self.x_N = len(xy0)

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
        self.x_f = torch.cat((temp_t, self.xy0), dim=1)
        for i in range(self.t_N - 1):
            temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[i + 1])
            x_f_temp = torch.cat((temp_t, self.xy0), dim=1)
            self.x_f = torch.cat((self.x_f, x_f_temp), dim=0)

        xy_N = int(np.sqrt(self.x_N))

        t_0 = np.linspace(self.lb[0], self.ub[0], self.t_N)[:, None]
        x_0 = np.linspace(self.lb[1], self.ub[1], xy_N)[:, None]
        y_0 = np.linspace(self.lb[2], self.ub[2], xy_N)[:, None]

        t_0_star, x_0_star = np.meshgrid(t_0, x_0)
        t_0_star = torch.from_numpy(t_0_star.flatten()[:, None])
        x_0_star = torch.from_numpy(x_0_star.flatten()[:, None])
        x_b1 = torch.cat((t_0_star, x_0_star, torch.full_like(torch.zeros_like(x_0_star), self.lb[2])),
                         dim=1)
        x_b2 = torch.cat((t_0_star, x_0_star, torch.full_like(torch.zeros_like(x_0_star), self.ub[2])),
                         dim=1)

        t_0_star, y_0_star = np.meshgrid(t_0, y_0)
        t_0_star = torch.from_numpy(t_0_star.flatten()[:, None])
        y_0_star = torch.from_numpy(y_0_star.flatten()[:, None])
        x_b3 = torch.cat((t_0_star, torch.full_like(torch.zeros_like(y_0_star), self.lb[1]), y_0_star),
                         dim=1)
        x_b4 = torch.cat((t_0_star, torch.full_like(torch.zeros_like(y_0_star), self.ub[1]), y_0_star),
                         dim=1)

        self.x_b = torch.cat((x_b1, x_b2, x_b3, x_b4), dim=0)
        self.x_b_u = self.exact_u(self.x_b)

        self.lb = torch.from_numpy(self.lb)
        self.ub = torch.from_numpy(self.ub)

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H) * x[:, [0]]

    def predict_U(self, x):
        return self.train_U(x)

    def hat_ut_and_Lu(self):
        x = Variable(self.x_f, requires_grad=True)

        u = self.train_U(x)
        d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        u_t = d[0][:, [0]]
        u_x = d[0][:, [1]]
        u_y = d[0][:, [2]]
        u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, [0]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [1]]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [2]]

        Lu = u_xx + u_yy + ((2 * x[:, [0]] ** (2 - alpha)) / (gamma(3 - alpha)) - 2 * x[:, [0]] ** 2) * torch.exp(
            x[:, [1]] + x[:, [2]])

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

        loss_b = torch.mean((self.train_U(self.x_b) - self.x_b_u) ** 2)
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

    order = 1
    layers = [3, 20, 20, 20, 20, 1]
    net = Net(layers)

    alpha = 0.85
    exact_u = lambda x: x[:, [0]] ** 2 * torch.exp(x[:, [1]] + x[:, [2]])

    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])

    # test
    t_pred_N = 100
    xy_pred_N = 100
    t_pred = np.linspace(lb[0], ub[0], t_pred_N)
    x_pred = np.linspace(lb[1], ub[1], xy_pred_N)[:, None]
    y_pred = np.linspace(lb[2], ub[2], xy_pred_N)[:, None]
    x_star, y_star = np.meshgrid(x_pred, y_pred)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    xy = np.hstack((x_star, y_star))
    xy = torch.from_numpy(xy)
    temp_t = torch.full_like(torch.zeros(xy_pred_N * xy_pred_N, 1), t_pred[0])
    x_test = torch.cat((temp_t, xy), dim=1)
    for i in range(t_pred_N - 1):
        temp_t = torch.full_like(torch.zeros(xy_pred_N * xy_pred_N, 1), t_pred[i + 1])
        x_test_temp = torch.cat((temp_t, xy), dim=1)
        x_test = torch.cat((x_test, x_test_temp), dim=0)
    x_test_exact = exact_u(x_test)

    t_N = 11
    xy_N = 11
    train_t = np.linspace(lb[0], ub[0], t_N)

    x_0 = np.linspace(lb[1], ub[1], xy_N)[:, None]
    y_0 = np.linspace(lb[2], ub[2], xy_N)[:, None]
    x_0_star, y_0_star = np.meshgrid(x_0, y_0)
    x_0_star = x_0_star.flatten()[:, None]
    y_0_star = y_0_star.flatten()[:, None]
    xy0 = np.hstack((x_0_star, y_0_star))

    xy0 = torch.from_numpy(xy0).float()
    u0 = xy0[:, [0]] * 0

    model = Model(
        order=order,
        net=net,
        xy0=xy0,
        u0=u0,
        t=train_t,
        lb=lb,
        ub=ub,
        x_test=x_test,
        x_test_exact=x_test_exact,
        exact_u=exact_u
    )

    model.train(epochs=3000)

    pred = model.predict_U(x_test).cpu().detach().numpy()
    pred = pred.reshape((t_pred_N, xy_pred_N, xy_pred_N))
    Exact = x_test_exact.cpu().detach().numpy()
    Exact = Exact.reshape((t_pred_N, xy_pred_N, xy_pred_N))

    nn = 200
    x_plot = np.linspace(lb[1], ub[1], nn)
    y_plot = np.linspace(lb[2], ub[2], nn)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

    u_data_pred = griddata(xy, pred[-1, :, :].reshape((-1, 1))[:, -1], (X_plot, Y_plot),
                           method='cubic')
    u_data_plot = griddata(xy, Exact[-1, :, :].reshape((-1, 1))[:, -1], (X_plot, Y_plot),
                           method='cubic')

    fig = plt.figure()
    ax = fig.add_subplot(121)
    h = ax.imshow(u_data_plot, interpolation='nearest', cmap='seismic',
                  extent=[lb[0], ub[1], lb[2], ub[2]],
                  origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Exact $u(x,t)$', fontsize=10)

    ax = fig.add_subplot(122)
    h = ax.imshow(u_data_pred, interpolation='nearest', cmap='seismic',
                  extent=[lb[0], ub[1], lb[2], ub[2]],
                  origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Pred $\\tilde{u}(x,t)$', fontsize=10)

    fig.tight_layout(h_pad=1)

    plt.savefig('TFADE_single.pdf')

    plt.show()


def show_NxNt():
    global alpha
    alpha = 0.85
    exact_u = lambda x: x[:, [0]] ** 2 * torch.exp(x[:, [1]] + x[:, [2]])

    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])

    layers = [3, 20, 20, 20, 20, 1]

    # test
    t_pred_N = 100
    xy_pred_N = 100
    t_pred = np.linspace(lb[0], ub[0], t_pred_N)
    x_pred = np.linspace(lb[1], ub[1], xy_pred_N)[:, None]
    y_pred = np.linspace(lb[2], ub[2], xy_pred_N)[:, None]
    x_star, y_star = np.meshgrid(x_pred, y_pred)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    xy = np.hstack((x_star, y_star))
    xy = torch.from_numpy(xy)
    temp_t = torch.full_like(torch.zeros(xy_pred_N * xy_pred_N, 1), t_pred[0])
    x_test = torch.cat((temp_t, xy), dim=1)
    for i in range(t_pred_N - 1):
        temp_t = torch.full_like(torch.zeros(xy_pred_N * xy_pred_N, 1), t_pred[i + 1])
        x_test_temp = torch.cat((temp_t, xy), dim=1)
        x_test = torch.cat((x_test, x_test_temp), dim=0)
    x_test_exact = exact_u(x_test)

    xy_N = 11
    x_0 = np.linspace(lb[1], ub[1], xy_N)[:, None]
    y_0 = np.linspace(lb[2], ub[2], xy_N)[:, None]
    x_0_star, y_0_star = np.meshgrid(x_0, y_0)
    x_0_star = x_0_star.flatten()[:, None]
    y_0_star = y_0_star.flatten()[:, None]
    xy0 = np.hstack((x_0_star, y_0_star))
    xy0 = torch.from_numpy(xy0).float()
    u0 = xy0[:, [0]] * 0

    orders = [0, 1, 2]
    t_N_all = np.array([6, 11, 21, 41])

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
                xy0=xy0,
                u0=u0,
                t=train_t,
                lb=lb,
                ub=ub,
                x_test=x_test,
                x_test_exact=x_test_exact,
                exact_u=exact_u
            )

            error, elapsed, loss = model.train(epochs=5000)
            error_all[i, j] = error

    xy_N = 21
    x_0 = np.linspace(lb[1], ub[1], xy_N)[:, None]
    y_0 = np.linspace(lb[2], ub[2], xy_N)[:, None]
    x_0_star, y_0_star = np.meshgrid(x_0, y_0)
    x_0_star = x_0_star.flatten()[:, None]
    y_0_star = y_0_star.flatten()[:, None]
    xy0 = np.hstack((x_0_star, y_0_star))
    xy0 = torch.from_numpy(xy0).float()
    u0 = xy0[:, [0]] * 0

    orders = [0, 1, 2]
    t_N_all = np.array([6, 11, 21, 41])

    error_all1 = np.zeros([len(t_N_all), len(orders)])

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
                xy0=xy0,
                u0=u0,
                t=train_t,
                lb=lb,
                ub=ub,
                x_test=x_test,
                x_test_exact=x_test_exact,
                exact_u=exact_u
            )

            error, elapsed, loss = model.train(epochs=5000)
            error_all1[i, j] = error

    plt.rc('legend', fontsize=16)
    fig = plt.figure(figsize=(10, 4))
    fig.add_subplot(1, 2, 1)

    plt.yscale('log')
    plt.xlabel('$M_t$', fontsize=20)
    plt.title('relative $L2$ error, $M_x = 11 \\times 11$', fontsize=20)
    plt.plot(t_N_all, error_all[:, 0], 'b-', marker="*", linewidth=2, label='p=1')
    plt.plot(t_N_all, error_all[:, 1], 'g-', marker="*", linewidth=2, label='p=3')
    plt.plot(t_N_all, error_all[:, 2], 'y-', marker="*", linewidth=2, label='p=5')
    plt.legend()
    plt.tight_layout()

    fig.add_subplot(1, 2, 2)
    plt.yscale('log')
    plt.xlabel('$M_t$', fontsize=20)
    plt.title('relative $L2$ error, $M_x = 31 \\times 31$', fontsize=20)
    plt.plot(t_N_all, error_all1[:, 0], 'b-', marker="*", linewidth=2, label='p=1')
    plt.plot(t_N_all, error_all1[:, 1], 'g-',marker="*",  linewidth=2, label='p=3')
    plt.plot(t_N_all, error_all1[:, 2], 'y-', marker="*", linewidth=2, label='p=5')
    plt.legend()
    plt.tight_layout()

    plt.savefig('TFADE_NxNt.pdf')
    plt.show()


def show_all():
    global alpha
    exact_u = lambda x: x[:, [0]] ** 2 * torch.exp(x[:, [1]] + x[:, [2]])

    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])

    layers = [3, 20, 20, 20, 20, 1]

    # test
    t_pred_N = 100
    xy_pred_N = 100
    t_pred = np.linspace(lb[0], ub[0], t_pred_N)
    x_pred = np.linspace(lb[1], ub[1], xy_pred_N)[:, None]
    y_pred = np.linspace(lb[2], ub[2], xy_pred_N)[:, None]
    x_star, y_star = np.meshgrid(x_pred, y_pred)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    xy = np.hstack((x_star, y_star))
    xy = torch.from_numpy(xy)
    temp_t = torch.full_like(torch.zeros(xy_pred_N * xy_pred_N, 1), t_pred[0])
    x_test = torch.cat((temp_t, xy), dim=1)
    for i in range(t_pred_N - 1):
        temp_t = torch.full_like(torch.zeros(xy_pred_N * xy_pred_N, 1), t_pred[i + 1])
        x_test_temp = torch.cat((temp_t, xy), dim=1)
        x_test = torch.cat((x_test, x_test_temp), dim=0)
    x_test_exact = exact_u(x_test)

    xy_N = 11
    x_0 = np.linspace(lb[1], ub[1], xy_N)[:, None]
    y_0 = np.linspace(lb[2], ub[2], xy_N)[:, None]
    x_0_star, y_0_star = np.meshgrid(x_0, y_0)
    x_0_star = x_0_star.flatten()[:, None]
    y_0_star = y_0_star.flatten()[:, None]
    xy0 = np.hstack((x_0_star, y_0_star))
    xy0 = torch.from_numpy(xy0).float()
    u0 = xy0[:, [0]] * 0

    alphas = [0.7, 0.8, 0.9]
    orders = [0, 1, 2]
    t_N_all = np.array([6, 11, 21, 41, 81])

    error_all = np.zeros([len(t_N_all), len(orders) * len(alphas)])

    for i in range(len(alphas)):
        alpha = alphas[i]
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
                    xy0=xy0,
                    u0=u0,
                    t=train_t,
                    lb=lb,
                    ub=ub,
                    x_test=x_test,
                    x_test_exact=x_test_exact,
                    exact_u=exact_u
                )
                error, elapsed, loss = model.train(epochs=5000)
                error_all[j, i * len(orders) + k] = error

    # np.savetxt('TFADE_all1.txt',error_all)

    file = open('TFADE_all.txt', 'w+')
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

    show_single()

    # show_NxNt()

    # show_all()
