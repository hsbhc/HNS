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

sys.path.append("../..")
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
    def __init__(self, order, net, xyz0, u0, t, lb, ub,
                 x_test, x_test_exact, exact_u
                 ):

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

    def make_boundary(self, t_0, x_0, y_0, z_0):

        t_0_star, x_0_star, y_0_star = np.meshgrid(t_0, x_0, y_0)
        t_0_star = torch.from_numpy(t_0_star.flatten()[:, None])
        x_0_star = torch.from_numpy(x_0_star.flatten()[:, None])
        y_0_star = torch.from_numpy(y_0_star.flatten()[:, None])
        z_b1 = torch.cat((t_0_star, x_0_star, y_0_star, torch.full_like(torch.zeros_like(x_0_star), self.lb[3])),
                         dim=1)
        z_b2 = torch.cat((t_0_star, x_0_star, y_0_star, torch.full_like(torch.zeros_like(x_0_star), self.ub[3])),
                         dim=1)

        t_0_star, x_0_star, z_0_star = np.meshgrid(t_0, x_0, z_0)
        t_0_star = torch.from_numpy(t_0_star.flatten()[:, None])
        x_0_star = torch.from_numpy(x_0_star.flatten()[:, None])
        z_0_star = torch.from_numpy(z_0_star.flatten()[:, None])
        y_b1 = torch.cat((t_0_star, x_0_star, torch.full_like(torch.zeros_like(x_0_star), self.lb[3]), z_0_star),
                         dim=1)
        y_b2 = torch.cat((t_0_star, x_0_star, torch.full_like(torch.zeros_like(x_0_star), self.ub[3]), z_0_star),
                         dim=1)

        t_0_star, y_0_star, z_0_star = np.meshgrid(t_0, y_0, z_0)
        t_0_star = torch.from_numpy(t_0_star.flatten()[:, None])
        y_0_star = torch.from_numpy(y_0_star.flatten()[:, None])
        z_0_star = torch.from_numpy(z_0_star.flatten()[:, None])
        x_b1 = torch.cat((t_0_star, torch.full_like(torch.zeros_like(x_0_star), self.lb[3]), y_0_star, z_0_star),
                         dim=1)
        x_b2 = torch.cat((t_0_star, torch.full_like(torch.zeros_like(x_0_star), self.ub[3]), y_0_star, z_0_star),
                         dim=1)

        self.x_b = torch.cat((x_b1, x_b2, y_b1, y_b2, z_b1, z_b2), dim=0)
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

        xyz_N = int(np.cbrt(self.x_N))

        t_0 = np.linspace(lb[0], ub[0], t_N)
        x_0 = np.linspace(lb[1], ub[1], xyz_N)
        y_0 = np.linspace(lb[2], ub[2], xyz_N)
        z_0 = np.linspace(lb[3], ub[3], xyz_N)

        self.make_boundary(t_0, x_0, y_0, z_0)

        self.lb = torch.from_numpy(self.lb)
        self.ub = torch.from_numpy(self.ub)

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H) * x[:, [0]] + exact_u0(x[:,1:4])

    def predict_U(self, x):
        return self.train_U(x)

    def hat_ut_and_Lu(self):
        x = Variable(self.x_f, requires_grad=True)
        u = self.train_U(x)
        d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        u_t = d[0][:, [0]]
        u_x = d[0][:, [1]]
        u_y = d[0][:, [2]]
        u_z = d[0][:, [3]]
        u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, [0]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [1]]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [2]]
        u_zz = torch.autograd.grad(u_z, x, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, [3]]

        Lu = u_xx + u_yy + u_zz - u_x - u_y - u_z + gamma(3) / gamma(3 - alpha) * x[:, [0]] ** (2-alpha) - (
                -torch.cos(x[:, [1]]) - torch.cos(x[:, [2]]) - torch.cos(x[:, [3]])) + (
                     -torch.sin(x[:, [1]]) - torch.sin(x[:, [2]]) - torch.sin(x[:, [3]]))

        u = u.reshape(self.t_N, -1)
        u_t = u_t.reshape(self.t_N, -1)
        u_tt = u_tt.reshape(self.t_N, -1)
        Lu = Lu.reshape(self.t_N, -1)

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


# def draw(t_num_index):
#     pred = model.predict_U(x_test).detach().numpy()
#     pred = pred.reshape((t_pred_N, xy_pred_N, xy_pred_N))
#     Exact = x_test_exact.detach().numpy()
#     Exact = Exact.reshape((t_pred_N, xy_pred_N, xy_pred_N))
#
#     nn = 200
#     x_plot = np.linspace(lb[1], ub[1], nn)
#     y_plot = np.linspace(lb[2], ub[2], nn)
#     X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
#
#     u_data_pred = griddata(xy, pred[t_num_index, :, :].reshape((-1, 1))[:, -1], (X_plot, Y_plot),
#                            method='cubic')
#     u_data_plot = griddata(xy, Exact[t_num_index, :, :].reshape((-1, 1))[:, -1], (X_plot, Y_plot),
#                            method='cubic')
#
#     fig = plt.figure()
#     ax = fig.add_subplot(121)
#     h = ax.imshow(u_data_plot, interpolation='nearest', cmap='seismic',
#                   extent=[lb[0], ub[1], lb[2], ub[2]],
#                   origin='lower')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#
#     fig.colorbar(h, cax=cax)
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$y$')
#     ax.set_title('Exact $u(x,t)$', fontsize=10)
#
#     ax = fig.add_subplot(122)
#     h = ax.imshow(u_data_pred, interpolation='nearest', cmap='seismic',
#                   extent=[lb[0], ub[1], lb[2], ub[2]],
#                   origin='lower')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#
#     fig.colorbar(h, cax=cax)
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$y$')
#     ax.set_title('Pred $\\tilde{u}(x,t)$', fontsize=10)
#
#     fig.tight_layout(h_pad=1)
#
#     plt.show()


if __name__ == '__main__':
    set_seed(1234)
    # train
    order = 0
    layers = [4, 20, 20, 20, 20, 1]
    net = Net(layers)

    alpha = 0.5
    exact_u = lambda x: x[:, [0]] ** 2 + torch.cos(x[:, [1]]) + torch.cos(x[:, [2]]) + torch.cos(x[:, [3]])
    exact_u0 = lambda x: torch.cos(x[:, [1]]) + torch.cos(x[:, [2]]) + torch.cos(x[:, [0]])

    lb = np.array([0.0, 0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0, 1.0])

    # test
    t_pred_N = 51
    xyz_pred_N = 51
    t_pred = np.linspace(lb[0], ub[0], t_pred_N)
    x_pred = np.linspace(lb[1], ub[1], xyz_pred_N)
    y_pred = np.linspace(lb[2], ub[2], xyz_pred_N)
    z_pred = np.linspace(lb[3], ub[3], xyz_pred_N)
    x_star, y_star, z_star = np.meshgrid(x_pred, y_pred, z_pred)
    x_star = x_star.flatten()
    y_star = y_star.flatten()
    z_star = z_star.flatten()
    xyz = np.stack((x_star, y_star, z_star), axis=1)
    xyz = torch.from_numpy(xyz)
    temp_t = torch.full_like(torch.zeros(xyz_pred_N ** 3, 1), t_pred[0])
    x_test = torch.cat((temp_t, xyz), dim=1)
    for i in range(t_pred_N - 1):
        temp_t = torch.full_like(torch.zeros(xyz_pred_N ** 3, 1), t_pred[i + 1])
        x_test_temp = torch.cat((temp_t, xyz), dim=1)
        x_test = torch.cat((x_test, x_test_temp), dim=0)
    x_test_exact = exact_u(x_test)

    t_N = 3
    xyz_N = 11
    train_t = np.linspace(lb[0], ub[0], t_N)

    x_0 = np.linspace(lb[1], ub[1], xyz_N)
    y_0 = np.linspace(lb[2], ub[2], xyz_N)
    z_0 = np.linspace(lb[3], ub[3], xyz_N)
    x_0_star, y_0_star, z_0_star = np.meshgrid(x_0, y_0, z_0)
    x_0_star = x_0_star.flatten()
    y_0_star = y_0_star.flatten()
    z_0_star = z_0_star.flatten()
    xyz0 = np.stack((x_0_star, y_0_star, z_0_star), axis=1)

    xyz0 = torch.from_numpy(xyz0).float()
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

    model.train(epochs=1000)
    #
    # draw_epoch_loss()
    # draw(t_num_index=-1)
