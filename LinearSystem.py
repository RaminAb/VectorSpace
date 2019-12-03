import VectorSpace as vs
import sympy as sym
import numpy as np
from numpy import math as mth
import scipy.linalg as la
import scipy.integrate as integrate

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
    
    return F_mat, F_Mat

def runsys(A,**kwargs):
    B = kwargs.get('B',np.zeros(A.shape))
    u = kwargs.get('u',np.zeros((A.shape[0],1)))
    show = kwargs.get('show',[''])
    method = kwargs.get('method','numerical')
    step = kwargs.get('step',0.1)
    time = kwargs['t']
    x0 = kwargs['x0']
    
    if method == 'numerical':
        sys = lambda x,t : np.squeeze(A.dot(np.array([x]).transpose())+B.dot(u))
        sol = integrate.odeint(sys,x0,time)
    elif method == 'exact':
        l = sym.Symbol('l')
        t = sym.Symbol('t')
        f = sym.exp(l*t)
        x0 = sym.Matrix(x0)
        f_A = sym.Matrix(MatFun(A,f,l)[0]) * x0
        solFun = lambda x : np.array(f_A.subs(t,x).evalf())
        sol = np.concatenate(list(map(solFun, time)),axis = 1).transpose()
    elif method == 'discrete':
        l = sym.Symbol('l')
        f = sym.exp(l*step)        
        Ad = MatFun(A,f,l)[0]
        sol = np.zeros((2,time.shape[0]))
        sol[:,0] = np.array([x0])
        for i in range(1,time.shape[0]):
            sol[:,i] = np.squeeze(Ad.dot(sol[:,i-1][:,np.newaxis]))
            
        sol = sol.transpose()
    if 'Jordan' in show:
        print(vs.Transform(A,'J')[1])
    elif 'Solution' in show :
        L = sym.Symbol('l')
        T = sym.Symbol('t')
        f = sym.exp(L*T)
        print(MatFun(A,f,L)[1])

    return sol
