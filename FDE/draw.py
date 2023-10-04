import numpy as np
from matplotlib import pyplot as plt

# 6  & 2.77e-01 & 2.60e-01 & 2.82e-03 & 2.64e-03 \\
# 11 & 8.90e-02 & 8.65e-02 & 4.45e-05 & 6.27e-05 \\
# 21 & 3.14e-02 & 3.10e-02 & 1.24e-05 & 1.22e-05 \\
# 41 & 1.13e-02 & 1.12e-02 & 5.88e-06 & 1.15e-04 \\
# 81 & 4.10e-03 & 4.10e-03 & 4.70e-06 & 9.94e-05 \\
# 101 & 2.90e-03 & 2.95e-03 & 4.63e-06 & 9.42e-05 \\
# 201 & 1.10e-03 & 1.10e-03 & 4.09e-06 & 8.46e-04 \\
# 401 & 3.78e-04 & 3.78e-04 & 3.97e-06 & 1.28e-02 \\
# 801 & 1.35e-04 & 1.37e-04 & 3.66e-06 & 6.80e-01 \\

M_t = [6, 11, 21, 41, 81, 101, 201, 401, 801]

FDM = [2.77e-01, 8.90e-02, 3.14e-02, 1.13e-02, 4.10e-03, 2.90e-03, 1.10e-03, 3.78e-04, 1.35e-04]
P1 = [2.60e-01, 8.65e-02, 3.10e-02, 1.12e-02, 4.10e-03, 2.95e-03, 1.10e-03, 3.78e-04, 1.37e-04]
P3 = [2.82e-03, 4.45e-05, 1.24e-05, 5.88e-06, 4.70e-06, 4.63e-06, 4.09e-06, 3.97e-06, 3.66e-06]
P5 = [2.64e-03, 6.27e-05, 1.22e-05, 1.15e-04, 9.94e-05, 9.42e-05, 8.46e-04, 1.28e-02, 6.80e-01]

plt.yscale('log')
plt.rc('legend', fontsize=16)
plt.xlabel('$M_t$', fontsize=20)
plt.ylabel('$L2error$', fontsize=20)
# plt.title('$Error$', fontsize=20)
plt.plot(M_t, FDM, 'b*-', label='FDM')
plt.plot(M_t, P1, 'r*-', label='$p=1$')
plt.plot(M_t, P3, 'y*-', label='$p=3$')
plt.plot(M_t, P5, 'g*-', label='$p=5$')
plt.legend()
plt.tight_layout()
plt.savefig('FDM_Comparison.pdf')
plt.show()