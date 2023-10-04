import numpy as np
import torch

from scipy.special import gamma
torch.set_default_dtype(torch.double)


def hermite_getH1(a, b, order=1):
    if order == 0:
        A = np.array(
            [[1, a], [1, b]])
        result = []
        for i in range(2):
            b = np.array([0, 0])
            b[i] = 1
            result.append(np.linalg.solve(A, b))
        return np.array(result)

    if order == 1:
        A = np.array(
            [[1, a, a ** 2, a ** 3], [1, b, b ** 2, b ** 3], [0, 1, a * 2, a ** 2 * 3], [0, 1, b * 2, b ** 2 * 3]])
        result = []
        for i in range(4):
            b = np.array([0, 0, 0, 0])
            b[i] = 1
            result.append(np.linalg.solve(A, b))
        return np.array(result)

    if order == 2:
        A = np.array(
            [[1, a, a ** 2, a ** 3, a ** 4, a ** 5], [1, b, b ** 2, b ** 3, b ** 4, b ** 5],
             [0, 1, a * 2, 3 * a ** 2, 4 * a ** 3, 5 * a ** 4], [0, 1, b * 2, 3 * b ** 2, 4 * b ** 3, 5 * b ** 4],
             [0, 0, 2, 6 * a, 12 * a ** 2, 20 * a ** 3], [0, 0, 2, 6 * b, 12 * b ** 2, 20 * b ** 3]])
        result = []
        for i in range(6):
            b = np.array([0, 0, 0, 0, 0, 0])
            b[i] = 1
            result.append(np.linalg.solve(A, b))
        return np.array(result)


def hermite_getH2(a, b, order=1):
    if order == 0:
        result = [[-b / (a - b), 1 / (a - b)],
                  [a / (a - b), -1 / (a - b)]
                  ]
        return np.array(result)

    if order == 1:
        result = [[-(-3 * a * b ** 2 + b ** 3) / (a - b) ** 3, -6 * a * b / (a - b) ** 3, 3 * (a + b) / (a - b) ** 3,
                   -2 / (a - b) ** 3],
                  [a ** 2 * (a - 3 * b) / (a - b) ** 3, 6 * a * b / (a - b) ** 3, -3 * (a + b) / (a - b) ** 3,
                   2 / (a - b) ** 3],
                  [-a * b ** 2 / (a - b) ** 2, -(-2 * a * b - b ** 2) / (a - b) ** 2, -(a + 2 * b) / (a - b) ** 2,
                   1 / (a - b) ** 2],
                  [-a ** 2 * b / (a - b) ** 2, -(-a ** 2 - 2 * a * b) / (a - b) ** 2, -(2 * a + b) / (a - b) ** 2,
                   1 / (a - b) ** 2],
                  ]
        return np.array(result)

    if order == 2:
        result = [
            [-(10 * a ** 2 * b ** 3 - 5 * a * b ** 4 + b ** 5) / (a - b) ** 5, 30 * a ** 2 * b ** 2 / (a - b) ** 5,
             -30 * a * b * (a + b) / (a - b) ** 5, 10 * (a ** 2 + 4 * a * b + b ** 2) / (a - b) ** 5,
             -15 * (a + b) / (a - b) ** 5, 6 / (a - b) ** 5],

            [a ** 3 * (a ** 2 - 5 * a * b + 10 * b ** 2) / (a - b) ** 5,
             -30 * a ** 2 * b ** 2 / (a - b) ** 5,
             30 * a * b * (a + b) / (a - b) ** 5, -10 * (a ** 2 + 4 * a * b + b ** 2) / (a - b) ** 5,
             15 * (a + b) / (a - b) ** 5, -6 / (a - b) ** 5],

            [a * (4 * a * b ** 3 - b ** 4) / (a - b) ** 4,
             -(12 * a ** 2 * b ** 2 + 4 * a * b ** 3 - b ** 4) / (a - b) ** 4,
             6 * a * b * (2 * a + 3 * b) / (a - b) ** 4, -2 * (2 * a ** 2 + 10 * a * b + 3 * b ** 2) / (a - b) ** 4,
             - (-7 * a - 8 * b) / (a - b) ** 4, -3 / (a - b) ** 4],

            [- (a ** 4 * b - 4 * a ** 3 * b ** 2) / (a - b) ** 4,
             -(-a ** 4 + 4 * a ** 3 * b + 12 * a ** 2 * b ** 2) / (a - b) ** 4,
             6 * a * (3 * a * b + 2 * b ** 2) / (a - b) ** 4,
             -2 * (3 * a ** 2 + 10 * a * b + 2 * b ** 2) / (a - b) ** 4,
             - (-8 * a - 7 * b) / (a - b) ** 4, -3 / (a - b) ** 4],

            [-a ** 2 * b ** 3 / (2 * (a - b) ** 3),
             a * (3 * a * b ** 2 + 2 * b ** 3) / (2 * (a - b) ** 3),
             -b * (3 * a ** 2 + 6 * a * b + b ** 2) / (2 * (a - b) ** 3),
             - (- a ** 2 - 6 * a * b - 3 * b ** 2) / (2 * (a - b) ** 3),
             - (2 * a + 3 * b) / (2 * (a - b) ** 3), 1 / (2 * (a - b) ** 3)],

            [a ** 3 * b ** 2 / (2 * (a - b) ** 3),
             -b * (2 * a ** 3 + 3 * a ** 2 * b) / (2 * (a - b) ** 3),
             a * (3 * b ** 2 + 6 * a * b + a ** 2) / (2 * (a - b) ** 3),
             - (b ** 2 + 6 * a * b + 3 * a ** 2) / (2 * (a - b) ** 3),
             (2 * b + 3 * a) / (2 * (a - b) ** 3), -1 / (2 * (a - b) ** 3)],
        ]
        return np.array(result)


def hermite_getAll(t_list, order=1):
    result = []
    for i in range(len(t_list) - 1):
        result.append(torch.from_numpy(hermite_getH2(t_list[i], t_list[i + 1], order)))
    result = torch.vstack(result)

    H_coefficient = []
    for i in range(order * 2 + 2):
        indices = torch.LongTensor([j for j in range(i, len(result), order * 2 + 2)])
        H_coefficient.append(torch.index_select(result, 0, indices))
    return H_coefficient


def D_u_0(H_coefficient, t_list, p, alpha):
    t_list = torch.from_numpy(t_list)
    p0 = p[0:-1]
    p1 = p[1:]
    first1 = H_coefficient[0][:, [1]] * p0
    first2 = H_coefficient[1][:, [1]] * p1
    first = first1 + first2
    ut = torch.zeros_like(p)
    for i in range(1, len(t_list)):
        tn = t_list[i]
        t1 = t_list[0:i]
        t2 = t_list[1:i + 1]
        start = first.T[:, 0:i] * (tn - t1) ** (1 - alpha) / (alpha - 1)
        end = first.T[:, 0:i] * (tn - t2) ** (1 - alpha) / (alpha - 1)
        ut[i] = torch.sum(end - start, dim=1) / gamma(1 - alpha)
    return ut


def D_u_1(H_coefficient, t_list, p, v, alpha):
    t_list = torch.from_numpy(t_list)
    p0 = p[0:-1]
    p1 = p[1:]
    v0 = v[0:-1]
    v1 = v[1:]
    first1 = H_coefficient[0][:, [1]] * p0
    first2 = H_coefficient[1][:, [1]] * p1
    first3 = H_coefficient[2][:, [1]] * v0
    first4 = H_coefficient[3][:, [1]] * v1

    second1 = H_coefficient[0][:, [2]] * p0
    second2 = H_coefficient[1][:, [2]] * p1
    second3 = H_coefficient[2][:, [2]] * v0
    second4 = H_coefficient[3][:, [2]] * v1

    third1 = H_coefficient[0][:, [3]] * p0
    third2 = H_coefficient[1][:, [3]] * p1
    third3 = H_coefficient[2][:, [3]] * v0
    third4 = H_coefficient[3][:, [3]] * v1

    first = first1 + first2 + first3 + first4
    second = second1 + second2 + second3 + second4
    third = third1 + third2 + third3 + third4

    ut = torch.zeros_like(p)
    for i in range(1, len(t_list)):
        tn = t_list[i]
        t1 = t_list[0:i]
        t2 = t_list[1:i + 1]
        a = third.T[:, 0:i]
        b = second.T[:, 0:i]
        c = first.T[:, 0:i]
        start = (tn - t1) ** (1 - alpha) * (
                c * (6 - 5 * alpha + alpha ** 2) - 2 * b * (alpha - 3) * (tn + t1 - alpha * t1) + 3 * a * (
                2 * tn ** 2 - 2 * (alpha - 1) * tn * t1 + (2 - 3 * alpha + alpha ** 2) * t1 ** 2)
        )

        end = (tn - t2) ** (1 - alpha) * (
                c * (6 - 5 * alpha + alpha ** 2) - 2 * b * (alpha - 3) * (tn + t2 - alpha * t2) + 3 * a * (
                2 * tn ** 2 - 2 * (alpha - 1) * tn * t2 + (2 - 3 * alpha + alpha ** 2) * t2 ** 2)
        )

        ut[i] = torch.sum(end - start, dim=1) / ((alpha - 1) * (alpha - 2) * (alpha - 3)) / gamma(1 - alpha)
    return ut


def D_u_2(H_coefficient, t_list, p, v, a, alpha):
    t_list = torch.from_numpy(t_list)
    p0 = p[0:-1]
    p1 = p[1:]
    v0 = v[0:-1]
    v1 = v[1:]
    a0 = a[0:-1]
    a1 = a[1:]
    first1 = H_coefficient[0][:, [1]] * p0
    first2 = H_coefficient[1][:, [1]] * p1
    first3 = H_coefficient[2][:, [1]] * v0
    first4 = H_coefficient[3][:, [1]] * v1
    first5 = H_coefficient[4][:, [1]] * a0
    first6 = H_coefficient[5][:, [1]] * a1

    second1 = H_coefficient[0][:, [2]] * p0
    second2 = H_coefficient[1][:, [2]] * p1
    second3 = H_coefficient[2][:, [2]] * v0
    second4 = H_coefficient[3][:, [2]] * v1
    second5 = H_coefficient[4][:, [2]] * a0
    second6 = H_coefficient[5][:, [2]] * a1

    third1 = H_coefficient[0][:, [3]] * p0
    third2 = H_coefficient[1][:, [3]] * p1
    third3 = H_coefficient[2][:, [3]] * v0
    third4 = H_coefficient[3][:, [3]] * v1
    third5 = H_coefficient[4][:, [3]] * a0
    third6 = H_coefficient[5][:, [3]] * a1

    fourth1 = H_coefficient[0][:, [4]] * p0
    fourth2 = H_coefficient[1][:, [4]] * p1
    fourth3 = H_coefficient[2][:, [4]] * v0
    fourth4 = H_coefficient[3][:, [4]] * v1
    fourth5 = H_coefficient[4][:, [4]] * a0
    fourth6 = H_coefficient[5][:, [4]] * a1

    fifth1 = H_coefficient[0][:, [5]] * p0
    fifth2 = H_coefficient[1][:, [5]] * p1
    fifth3 = H_coefficient[2][:, [5]] * v0
    fifth4 = H_coefficient[3][:, [5]] * v1
    fifth5 = H_coefficient[4][:, [5]] * a0
    fifth6 = H_coefficient[5][:, [5]] * a1

    first = first1 + first2 + first3 + first4 + first5 + first6
    second = second1 + second2 + second3 + second4 + second5 + second6
    third = third1 + third2 + third3 + third4 + third5 + third6
    fourth = fourth1 + fourth2 + fourth3 + fourth4 + fourth5 + fourth6
    fifth = fifth1 + fifth2 + fifth3 + fifth4 + fifth5 + fifth6

    ut = torch.zeros_like(p)
    for i in range(1, len(t_list)):
        tn = t_list[i]
        t1 = t_list[0:i]
        t2 = t_list[1:i + 1]
        a = fifth.T[:, 0:i]
        b = fourth.T[:, 0:i]
        c = third.T[:, 0:i]
        d = second.T[:, 0:i]
        e = first.T[:, 0:i]
        f = alpha

        start = (tn - t1) ** (1 - alpha) * (
                e * (120 - 154 * f + 71 * f ** 2 - 14 * f ** 3 + f ** 4) +
                120 * c * tn ** 2 - 54 * c * f * tn ** 2 + 6 * c * f ** 2 * tn ** 2 + 120 * b * tn ** 3 - 24 * b * f * tn ** 3 +
                120 * a * tn ** 4 + 120 * c * tn * t1 - 174 * c * f * tn * t1 + 60 * c * f ** 2 * tn * t1 - 6 * c * f ** 3 * tn * t1 +
                120 * b * tn ** 2 * t1 - 144 * b * f * tn ** 2 * t1 + 24 * b * f ** 2 * tn ** 2 * t1 + 120 * a * tn ** 3 * t1 - 120 * a * f * tn ** 3 * t1 +
                120 * c * t1 ** 2 - 234 * c * f * t1 ** 2 + 147 * c * f ** 2 * t1 ** 2 - 36 * c * f ** 3 * t1 ** 2 + 3 * c * f ** 4 * t1 ** 2 + 120 * b * tn * t1 ** 2 -
                204 * b * f * tn * t1 ** 2 + 96 * b * f ** 2 * tn * t1 ** 2 - 12 * b * f ** 3 * tn * t1 ** 2 + 120 * a * tn ** 2 * t1 ** 2 - 180 * a * f * tn ** 2 * t1 ** 2 +
                60 * a * f ** 2 * tn ** 2 * t1 ** 2 + 120 * b * t1 ** 3 - 244 * b * f * t1 ** 3 + 164 * b * f ** 2 * t1 ** 3 - 44 * b * f ** 3 * t1 ** 3 + 4 * b * f ** 4 * t1 ** 3 +
                120 * a * tn * t1 ** 3 - 220 * a * f * tn * t1 ** 3 + 120 * a * f ** 2 * tn * t1 ** 3 - 20 * a * f ** 3 * tn * t1 ** 3 + 120 * a * t1 ** 4 - 250 * a * f * t1 ** 4 +
                175 * a * f ** 2 * t1 ** 4 - 50 * a * f ** 3 * t1 ** 4 + 5 * a * f ** 4 * t1 ** 4 - 2 * d * (
                        -60 + 47 * f - 12 * f ** 2 + f ** 3) * (tn + t1 - f * t1)
        )

        end = (tn - t2) ** (1 - alpha) * (
                e * (120 - 154 * f + 71 * f ** 2 - 14 * f ** 3 + f ** 4) +
                120 * c * tn ** 2 - 54 * c * f * tn ** 2 + 6 * c * f ** 2 * tn ** 2 + 120 * b * tn ** 3 - 24 * b * f * tn ** 3 +
                120 * a * tn ** 4 + 120 * c * tn * t2 - 174 * c * f * tn * t2 + 60 * c * f ** 2 * tn * t2 - 6 * c * f ** 3 * tn * t2 +
                120 * b * tn ** 2 * t2 - 144 * b * f * tn ** 2 * t2 + 24 * b * f ** 2 * tn ** 2 * t2 + 120 * a * tn ** 3 * t2 - 120 * a * f * tn ** 3 * t2 +
                120 * c * t2 ** 2 - 234 * c * f * t2 ** 2 + 147 * c * f ** 2 * t2 ** 2 - 36 * c * f ** 3 * t2 ** 2 + 3 * c * f ** 4 * t2 ** 2 + 120 * b * tn * t2 ** 2 -
                204 * b * f * tn * t2 ** 2 + 96 * b * f ** 2 * tn * t2 ** 2 - 12 * b * f ** 3 * tn * t2 ** 2 + 120 * a * tn ** 2 * t2 ** 2 - 180 * a * f * tn ** 2 * t2 ** 2 +
                60 * a * f ** 2 * tn ** 2 * t2 ** 2 + 120 * b * t2 ** 3 - 244 * b * f * t2 ** 3 + 164 * b * f ** 2 * t2 ** 3 - 44 * b * f ** 3 * t2 ** 3 + 4 * b * f ** 4 * t2 ** 3 +
                120 * a * tn * t2 ** 3 - 220 * a * f * tn * t2 ** 3 + 120 * a * f ** 2 * tn * t2 ** 3 - 20 * a * f ** 3 * tn * t2 ** 3 + 120 * a * t2 ** 4 - 250 * a * f * t2 ** 4 +
                175 * a * f ** 2 * t2 ** 4 - 50 * a * f ** 3 * t2 ** 4 + 5 * a * f ** 4 * t2 ** 4 - 2 * d * (
                        -60 + 47 * f - 12 * f ** 2 + f ** 3) * (tn + t2 - f * t2)
        )

        ut[i] = torch.sum(end - start, dim=1) / (
                (alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4) * (alpha - 5)) / gamma(1 - alpha)
    return ut
