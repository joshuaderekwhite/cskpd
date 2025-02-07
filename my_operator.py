# M always means matrix
import numpy as np
import pandas as pd
import itertools
import numpy.linalg as la

def Rearrange(C,p,d):
    p, d = np.array(p), np.array(d)
    assert C.ndim > 1, "The size of the tensor must be of dimensions of 2 or more."
    assert C.ndim == len(p) == len(d), "The dimension size of C, and the lengths of p and d must all be equal"
    assert (C.shape == p*d).all(), "The dimensions of C, must be equal to the product of each element of p*d"
    slices = []
    RC = []
    for dim in range(C.ndim):
        slices.append([])
        for i in range(p[dim]):
            slices[dim].append(slice(d[dim]*i, d[dim]*(i+1)))
    RC = [C[s].reshape(-1,1) for s in list(itertools.product(*slices))]
    return np.concatenate(RC, axis=1).T

def func_kron_ab(a_hat,b_hat,R,p,d):
    Ra_hat = a_hat.reshape(R,-1)
    Rb_hat = b_hat.reshape(R,-1)
    A = []
    B = []
    kron_ab = []

    Ra_hat = a_hat.reshape(R,-1)
    Rb_hat = b_hat.reshape(R,-1)
    A = []
    B = []
    kron_ab = []
    for i in range(1,R+1):
        locals()['a_hat' + str(i)] = Ra_hat[i-1,:].reshape(-1,1)
        locals()['b_hat' + str(i)] = Rb_hat[i-1,:].reshape(-1,1)
        A.append(eval('a_hat' + str(i)).reshape(*p))
        B.append(eval('b_hat' + str(i)).reshape(*d))

    kron_ab = [np.kron(A[i],B[i]) for i in range(R)]
    beta_hat = sum(kron_ab)
    kron_ab.append(beta_hat)

    return A,B,kron_ab