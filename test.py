import numpy as np
from ttopt import TTOpt
from ttopt.ttopt_raw import ttopt_init, _iter

from teneva import ind_qtt_to_tt


def f(I):
    res = I.sum(axis=1)
    return res

def generate_J_with_k_zeros(d, k, r):
    J0 = [None for _ in range(d+1)]
    is_zero = np.array([True] * k + [False] * (d - k))
    np.random.shuffle(is_zero)

    J0[1] = np.ones((r, 1), dtype=int) * is_zero[0]

    for k in range(1, d-1):
        ir = np.ones((r, 1), dtype=int) * is_zero[k]
        J0[k+1] = np.hstack((J0[k], ir))

    print([i.shape for i in J0[1:-1]])

    for k in range(0, d-1):
        J0[k+1] = J0[k+1][:min(n**(k+1), r), :]

    print([i.shape for i in J0[1:-1]])

    return J0

def generate_random_J(n, rank, d):
    Jg_list = [np.reshape(np.arange(k), (-1, 1)) for k in n]

    Y0, r = ttopt_init(n, rank, Y0=None, seed=34, with_rank=True)
    J_list = [None] * (d + 1)
    for i in range(d - 1):
        J_list[i+1] = _iter(Y0[i], J_list[i], Jg_list[i], uint8=False, l2r=True)

    return J_list
 
d = int(1e3)
N = 5
rank = 2
evals = 10000

n = np.ones(d, dtype=int) * int(N)

# J_list = generate_random_J(n, rank, d)

# print([i.shape for i in J_list[1:-1]])
# print(J_list)

# J_merged = _merge_inds(J_list, n)
# J_unmerged = _unmerge_inds(J_merged, n)

# print(J_merged)
# print(J_unmerged)

tto = TTOpt(
        f=f,
        d=d,                 # Number of function dimensions
        n=N,                 # Number of grid points (number or list of len d)
        evals=evals,         # Number of function evaluations
        is_func=False,
        with_log=True,
        with_cache=False,
        uint8=True,
        packbits=True)


tto.optimize(rank)
print('DONE !')
