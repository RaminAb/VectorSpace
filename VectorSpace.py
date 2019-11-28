import sympy as sym
import numpy as np
import scipy.linalg as la
import re

#################    
# Classes ################################################################
#################

class linMap:
    def __init__(self,fun,V,W):
        self.fun = fun
        self.V = V
        self.W = W
        self.dimension = (getD(V),getD(W))
    def __call__(self,v):
        return vector(self.fun(v.vec),self.W)
    def null(self):
        return [invVec(M,basis(self.V)) for M in Mat(self,basis(self.V),basis(self.W)).nullspace()]

class operator(linMap):
    def __init__(self,fun,V):
        super().__init__(fun,V,V)

class vector:
    def __init__(self,array,space):
        if re.match(r'F.',space):
            self.vec = np.array(array)
        if re.match(r'C.',space):
            self.vec = array
        self.space = space
        self.dimension = getD(space)
    def __mul__(self,scalar):
        return vector(self.vec*scalar,self.space)
    __rmul__=__mul__
    def __truediv__(self,scalar):
        return self*(1/scalar)
    def __add__(self,other):
        return vector(self.vec+other.vec,self.space)
    def __sub__(self,other):
        return vector(self.vec-other.vec,self.space)
    def __str__(self):
        return '{}'.format(self.vec)
    def __repr__(self):
        return str(self)
    def innerproduct(self,other):
        if re.match(r'F.',self.space):
            return sum(self.vec * other.vec)
        if re.match(r'C.',self.space):
            return sym.integrate(self.vec*other.vec,(sym.Symbol('x'),-1,1))
    def norm(self):
        return sym.sqrt(self.innerproduct(self))
    def normalize(self):
        return self/self.norm()
    def initial(self):
        return vector(self.vec-self.vec,self.space)

    
#################    
# Functions ################################################################
#################
        
def vectors(array,space):
    return list(map(lambda x: vector(x,space) , array))


def zerov(space):
    if re.match(r'F.',space):
        return vector([0]*getD(space),space)
    if re.match(r'C.',space):
        return vector([0],space) 

def eye(space):
    n = getD(space)
    I = invMat(sym.eye(n),basis(space),basis(space))
    return I

def basis(space):
    n = getD(space)
    if re.match(r'F.',space):
        base = sym.Matrix.diag([1]*n)
        return vectors([base.row(i)[:] for i in range(n)],space)
    if re.match(r'C.',space):
        x = sym.Symbol('x')
        return vectors([x**i for i in range(n)],space)

def Gramm(base):
    eBase = base.copy()
    eBase[0] = base[0].normalize()
    for i in range(1,len(base)):
        eBase[i] = (base[i]- sum([base[i].innerproduct(eBase[j])*eBase[j]\
                    for j in range(i)],base[0].initial())).normalize()
    return eBase

def Vec(vec,base):
    n = len(base)
    eBase = Gramm(basis(vec.space))
    X = sym.symbols('x0:%d'%n)
    Eq = sym.zeros(1,n)
    for k in range(n):
        Eq[k] = sum([X[i]*(base[i].innerproduct(eBase[k])) for i in range(n)]) - vec.innerproduct(eBase[k])
    
    sol = sym.solve(Eq,X)
    return sym.Matrix([sol[X[i]] for i in range(n)])

def invVec(mat,base):
    if mat != []:
        return vector(sum([mat[i]*base[i].vec for i in range(len(base))]),base[0].space)
    else:
        return zerov(base[0].space,len(base))

def Mat(lMap,vBase,wBase):
    return sym.Matrix.hstack(*[Vec(lMap(VBase),wBase) for VBase in vBase])

def invMat(mat,vBase,wBase):
    def F(x):
        vx = vector(x,vBase[0].space)
        x_vec = Vec(vx,vBase)
        return (invVec(mat*x_vec,wBase)).vec
    if vBase[0].space == wBase[0].space:
        return operator(F,vBase[0].space)
    else:    
        return linMap(F,vBase[0].space,wBase[0].space)
    
def getD(space):
    return int(re.search(r'\d',space).group())
    
def Adj(linMap):
    vBase = basis(linMap.V)
    wBase = basis(linMap.W)
    return invMat(sym.Matrix.adjoint(Mat(linMap,vBase,wBase)),wBase,vBase)

def U(vBase,wBase):
    return Mat(eye(vBase[0].space),vBase,wBase)

def isindep(bList):
    M = sym.Matrix.hstack(*list(map(lambda x : sym.Matrix(x.vec),bList)))
    if  M.det() == 0:
        return 0
    else:
        return 1


#################    
# Scipy ################################################################
#################
eps = 1e-10

def makebasis(arr,n):
    basis = arr[:,0][:,np.newaxis]
    i = 1
    while basis.shape[1] < n:
        cmat = np.concatenate([basis,arr[:,i][:,np.newaxis]],axis=1)
        if np.abs(np.linalg.det(cmat.transpose().dot(cmat))) > eps :
            basis = cmat
        i = i+1
    return basis
    
def realize(mat):
    mat.real[np.absolute(np.real(mat))< eps]=0
    if np.array_equal(np.real(mat),mat):
        return np.real(mat)
    else:
        mat.imag[np.absolute(np.imag(mat))< eps]=0
    return mat

def gen_eigenvects(mat):
    Eigen = la.eig(mat)
    eigval = np.unique(np.around(Eigen[0],15))
    gen_eigvec = {}
    gen_eigval = {}
    for ev in eigval:
        eigvec = []
        nil = mat - ev*np.eye(mat.shape[0])
        alg_mlt = len(np.where(np.abs(Eigen[0]-ev)<eps)[0])
        geo_mlt = la.null_space(nil).shape[1]
        for i in range(alg_mlt):
            G = la.null_space(la.fractional_matrix_power(nil,i+1))
            eigvec.append(G)
            if G.shape[1] == alg_mlt:
                gen_eigval[ev] = (alg_mlt,geo_mlt,i+1)
                break
        gen_eigvec[ev] = makebasis(realize(np.concatenate(eigvec,axis=1)),alg_mlt)
    return gen_eigvec, gen_eigval

def JordanBasis(mat):
    Evec, Eval = gen_eigenvects(mat)
    Jb = {}
    for ev in Eval.keys():
        v = Evec[ev][:,-1][:,np.newaxis]
        Jb[ev] = []
        for i in range(Eval[ev][2]):
            Jb[ev].append(np.linalg.matrix_power(mat-ev*np.eye(mat.shape[0]),i).dot(v))
        Jb[ev] = realize(np.concatenate(Jb[ev],axis=1))[:,::-1]
    return np.concatenate(list(Jb.values()),axis = 1)

def diagonalize(mat):
    Evec, Eval = gen_eigenvects(mat)
    U = np.concatenate(list(Evec.values()),axis=1)
    Mat = la.inv(U).dot(mat).dot(U)
    return realize(Mat)

def Jordan(mat):
    U = JordanBasis(mat)
    Mat = la.inv(U).dot(mat).dot(U)
    return realize(Mat)

