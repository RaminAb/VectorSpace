import VectorSpace as vs
import sympy as sym
import numpy as np
from numpy import math as mth
import scipy.linalg as la

def MatFun(mat,f,x):
    U,mat_hat = vs.Jordan(mat)
    M = {}
    for ev in np.diagonal(mat_hat):
        num = len(np.where(np.diagonal(mat_hat)==ev)[0])
        M[ev] = sym.zeros(num)
        for k in range(num):
            F = (sym.diff(f,x,k)/mth.factorial(k)).subs(x,ev)
            M[ev][:num-k,k:] += sym.diag(*[F]*(num-k))

    return U.dot(la.block_diag(*list(M.values()))).dot(la.inv(U))
