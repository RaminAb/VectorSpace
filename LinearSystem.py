import VectorSpace as vs
import sympy as sym
import numpy as np
from numpy import math as mth
import scipy.linalg as la

def MatFun(f,mat):
    U,A_hat = sym.Matrix(mat).jordan_form()
    Evec, Eval = vs.eig(mat)
    M = {}
    for ev in np.diagonal(A_hat):
        num = Eval[ev]
        M[ev] = np.zeros((num[0],num[0]))
        for k in range(num[0]):
            F = (sym.diff(f,*f.free_symbols,k)/mth.factorial(k)).subs(*f.free_symbols,ev)
            np.fill_diagonal(M[ev][:,k:],F)

    return la.block_diag(*list(M.values()))
