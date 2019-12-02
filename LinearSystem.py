import VectorSpace as vs
import sympy as sym
import numpy as np
from numpy import math as mth
import scipy.linalg as la

def MatFun(mat,f,x):
    idx = [0]
    U, Mat = vs.Transform(mat,'J')
    for i in range(1,Mat.shape[1]):
        if Mat[i-1,i] == 0:
            idx.append(i)
    idx.append(Mat.shape[1])
    block_size = np.diff(idx)
    Block = []
    for i in range(block_size.shape[0]):
        F = lambda ev,k: (sym.diff(f,x,k)/mth.factorial(k)).subs(x,ev)
        F_M = sym.Matrix.diag(*[F(Mat[i,i],0)]*block_size[i])
        for k in range(1,block_size[i]):
            hpad = sym.Matrix(0,k,[])
            vpad = sym.Matrix(k,0,[])
            F_M = F_M + sym.Matrix.diag(hpad,*[F(Mat[i,i],k)]*(block_size[i]-k),vpad)
        Block.append(F_M)
    F_Mat = sym.Matrix.diag(*Block)
    F_mat = U.dot(F_Mat).dot(la.inv(U))
    
    return F_mat
# 
