"""
Credit : Ramin Abbasi
Update : 08/17/2020
========================
Vector Space (VS) Package
========================
This package includes tools from linear algebra to work on linear maps and
matrices. It requires sympy, numpy, scipy, and re packages. 


Classes
=======
VS includes two main classes:
    1) linMap (and operator)
    2) vector

linMap
------
It generates a linear map object (V -> W) with three attributes:
    1) fun: fun: sets the functionality of linear map
    2) V: str: sets the vector space V (e.g F3,P2)
    3) W: str: sets the vector space W (e.g F5,P1)
and the following methods:
    1) prod : to multiply two linMaps
    2) null : returns the null of a linMap
    3) eig  : returns the eigenval, eigenvec of linMap
    
vector
------  
It generates a vector object with two attributes:
    1) vec: array:  sets the array from which the vector is built (e.g [1,2,3])
    2) space: str:  sets the space in which the vector is built (e.g F3)
and the following methods:
    1) innerproduct : returns the innerproduct of two vectors (standard def)
    2) norm : returns the norm of a vector based on defined innerproduct
    3) normalize : normalizes a vector
    4) initial : sets the vector to zero

    
Functions on Linear Maps
========================

vectors(array,space) - generate a list of vectors from 'array' in 'space'
zerov(space) - generates a zero vector in 'space'
eye(space) - generate identity map on 'space'
basis(space) - generate standard basis on 'space'
Gramm(base) - generate an orthonormal basis from 'base' using Gram-Shmidt
Matv(vec,base) - returns the matrix of 'vec' using 'base'
InvMatv(mat,base) - returns the vector of 'mat' using 'base'
Mat(lMap,vBase,wBase) - returns the matrix of 'lMap' using two bases 'vBase','wBase'
invMat(mat,vBase,wBase) - returns the linMap of 'mat' using two bases 'vBase','wBase'
getD(space) - returns the dimension of 'space'
Adj(linMap) - returns the adjoint of 'linMap'
U(vBase,wBase) - returns the Unitary transformation from 'vBase' to 'wBase'
isindep(bList) - checks the linear independence of 'bList'
sym2num(Mat) - turns the sympy matrix into numpy array

Functions on Matrices
=====================


Public methods
--------------
eig(mat,mode) - calculates eigenvalues and generalized eigenvectors  

    Inputs: (mat,mode)
    -------
    mat  - input matrix
    mode - specifies the structure (D = Diagonal, J = Jordan)

    output: (gen_eigvec,gen_eigval)
    -------
    gen_eigval : returns 'a : (b,c,d)', where
        a : eigenvalue of 'mat'
        b : algebraic multiplicity of a
        c : geometric multiplicity of a
        d : the power of (z-a) in minimum polynomial
    gen_eigvec : returns '{a : array}'
        a : eigenvalue of 'mat'
        array : generalized eigenvectors of a
    

Transform(mat,form,field) - Transforms the matrix into a desired form
    
    Input: (mat,form,field)
    ------
    mat  : input matrix
    form : specifies the required form (D = Diagonal, J = Jordan)
    field: specifies the working field (C = Complex, R = Real)

    Output: (U,Mat)
    U : transformation matrix
    Mat : transformed matrix

Private methods
---------------
_setR(gen_eigvec,gen_eigval) - restricts the basis computation to Field R
_GenEigen(Base,nil,num,mode) - returns the correct basis for D or J form
_makebasis(arr,n) - rearranges the basis to have a uniform structure
_realize(mat) - turns a complex matrix 'mat' into real if possible
"""



import sympy as sym
import numpy as np
import scipy.linalg as la
import re
sym.init_printing(pretty_print=False)

#========================================================================= 
# Classes
#========================================================================= 

class linMap:
    def __init__(self,fun,V,W):
        self.fun = fun
        self.V = V
        self.W = W
    def __call__(self,v):
        return vector(self.fun(v.vec),self.W)
    def __mul__(self,scalar):
        return linMap(lambda x: scalar*self.fun(x),self.V,self.W)
    __rmul__=__mul__
    def __truediv__(self,scalar):
        return linMap(lambda x: self.fun(x)/scalar,self.V,self.W)
    def __add__(self,other):
        return linMap(lambda x: self.fun(x)+other.fun(x),self.V,self.W)
    def __sub__(self,other):
        return linMap(lambda x: self.fun(x)-other.fun(x),self.V,self.W)
    def __str__(self):
        if re.match(r'F.',self.V):
            n = getD(self.V)
            z = sym.symbols('z0:%d'%n)
            return '{}({}->{}) : {} -> {}'.format(type(self).__name__,self.V,self.W,z, self.fun(z))
        if re.match(r'P.',self.V):
            return '{}({}->{}) : {}'.format(type(self).__name__,self.V,self.W,self.fun.__doc__)
    def __repr__(self):
        return str(self)
    def prod(self,other):
        return linMap(lambda x: self.fun(other.fun(x)),self.V,other.W)
    def null(self):
        return [invMatv(M,basis(self.V)) for M in Mat(self,basis(self.V),basis(self.W)).nullspace()]
    def eig(self):
        M = Mat(self,basis(self.V),basis(self.W)).eigenvects()
        mat = {}
        for v in M:
            mat[v[0]] = invMatv(v[2][0],basis(self.V))
        return mat

class operator(linMap):
    def __init__(self,fun,V):
        super().__init__(fun,V,V)

class vector:
    def __init__(self,array,space):
        if re.match(r'F.',space):
            self.vec = np.array(array)
        if re.match(r'P.',space):
            self.vec = array
        self.space = space
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
        if re.match(r'P.',self.space):
            return sym.integrate(self.vec*other.vec,(sym.Symbol('x'),-sym.pi,sym.pi))
    def norm(self):
        return sym.sqrt(self.innerproduct(self))
    def normalize(self):
        return self/self.norm()
    def initial(self):
        return vector(self.vec-self.vec,self.space)

    
#========================================================================= 
# Linear Map Functions
#========================================================================= 
        
def vectors(array,space):
    return list(map(lambda x: vector(x,space) , array))


def zerov(space):
    if re.match(r'F.',space):
        return vector([0]*getD(space),space)
    if re.match(r'P.',space):
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
    if re.match(r'P.',space):
        x = sym.Symbol('x')
        return vectors([x**i for i in range(n)],space)

def Gramm(base):
    eBase = base.copy()
    eBase[0] = base[0].normalize()
    for i in range(1,len(base)):
        eBase[i] = (base[i]- sum([base[i].innerproduct(eBase[j])*eBase[j]\
                    for j in range(i)],base[0].initial())).normalize()
    return eBase

def Matv(vec,base):
    n = len(base)
    eBase = Gramm(basis(vec.space))
    X = sym.symbols('x0:%d'%n)
    Eq = sym.zeros(1,n)
    for k in range(n):
        Eq[k] = sum([X[i]*(base[i].innerproduct(eBase[k])) for i in range(n)]) - vec.innerproduct(eBase[k])
    
    sol = sym.solve(Eq,X)
    return sym.Matrix([sol[X[i]] for i in range(n)])

def invMatv(mat,base):
    if mat != []:
        return vector(sum([mat[i]*base[i].vec for i in range(len(base))]),base[0].space)
    else:
        return zerov(base[0].space)

def Mat(lMap,vBase,wBase):
    return sym.Matrix.hstack(*[Matv(lMap(VBase),wBase) for VBase in vBase])

def invMat(mat,vBase,wBase):
    def F(x):
        ''' Unknown '''
        vx = vector(x,vBase[0].space)
        x_vec = Matv(vx,vBase)
        return (invMatv(mat*x_vec,wBase)).vec
    if vBase[0].space == wBase[0].space:
        return operator(F,vBase[0].space)
    else:
        return linMap(F,vBase[0].space,wBase[0].space)
    
def getD(space):
    if re.match(r'F.',space):
        return int(re.search(r'\d',space).group())
    if re.match(r'P.',space):
        return int(re.search(r'\d',space).group())+1
    
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
    
def sym2num(Mat):
    return np.array(Mat).astype(np.float64)


#========================================================================= 
# Matrix Functions
#========================================================================= 
eps = 1e-10

def eig(mat,mode = 'D'):
    Eigen = la.eig(mat)
#    eigval = np.sort(np.unique(Eigen[0]))[::-1]
    eigval = np.sort(np.unique(np.around(Eigen[0],decimals = 15)))[::-1]
    gen_eigvec = {}
    gen_eigval = {}
    for ev in eigval:
        eigvec = []
        nil = mat - ev*np.eye(mat.shape[0])
        alg_mlt = len(np.where(np.abs(Eigen[0]-ev)<eps)[0])
        geo_mlt = la.null_space(nil,rcond = eps).shape[1]
        for i in range(alg_mlt):
            G = la.null_space(_realize(np.linalg.matrix_power(nil,i+1)),rcond = eps)
            eigvec.append(G)
            if G.shape[1] == alg_mlt:
                gen_eigval[ev] = (alg_mlt,geo_mlt,i+1)
                break
        base = _makebasis(_realize(np.concatenate(eigvec,axis=1)),alg_mlt)
        gen_eigvec[ev] = _GenEigen(base,nil,gen_eigval[ev],mode)
    return gen_eigvec, gen_eigval

def _setR(gen_eigvec,gen_eigval):
    num = 1
    for ev in gen_eigvec.keys():
        print(ev.imag)
        if not(abs(ev.imag-0)<eps):
            if num == 1:
                gen_eigvec[ev] = 2*gen_eigvec[ev].real
                num = num + 1
            else:
                gen_eigvec[ev] = 2*gen_eigvec[ev].imag
                num = num + 1
        if num == 3: num = 1
    return gen_eigvec, gen_eigval

def _GenEigen(Base,nil,num,mode):

    alg_mlt, geo_mlt, nil_pwr = num
    vList = np.hsplit(Base,Base.shape[1])
    idx = []
    G = {}
    J = []
    for i in range(len(vList)):
        for n in range(1,nil_pwr+1):
            w = np.linalg.matrix_power(nil,n).dot(vList[i])
            if (_realize(w) == 0).all():
                idx.append((i,n))
                break
    idx_eig = [i for i, x in enumerate(idx) if x[1] == 1]
    
    for indice in idx_eig:
        G[indice] = [vList[indice]]
        for i,v in enumerate(vList):
            w = np.linalg.matrix_power(nil,idx[i][1]-1).dot(v)
            cmat = np.concatenate([vList[indice],w],axis = 1)
            det = np.abs(np.linalg.det(cmat.transpose().dot(cmat)))
            if (det < eps) and idx[i][1]!=1 :
                G[indice].append(vList[i])
        G[indice] = np.concatenate(G[indice],axis=1)
        
    for i in G.keys():
        for j in range(G[i].shape[1]):
            Null = (np.linalg.matrix_power(nil,j).dot(G[i]))
            J.append(Null[:,-1][:,np.newaxis])
    J_Base = _realize(np.concatenate(J[::-1],axis=1))
    D_Base = np.concatenate(list(G.values()),axis = 1)
    if mode == 'J':
        return J_Base
    elif mode == 'D':
        return D_Base
    else:
        return D_Base
       
def _makebasis(arr,n): 
    base = arr[:,min(np.where((arr != 0))[1])][:,np.newaxis]
    i = 1
    while base.shape[1] < n:
        cmat = np.concatenate([base,arr[:,i][:,np.newaxis]],axis=1)
        if np.abs(np.linalg.det(cmat.transpose().dot(cmat))) > eps :
            base = cmat
        i = i+1
    return base
    
def _realize(mat):
    mat.real[np.absolute(np.real(mat))< eps]=0
    if np.array_equal(np.real(mat),mat):
        return np.real(mat)
    else:
        mat.imag[np.absolute(np.imag(mat))< eps]=0
    return mat

def Transform(mat,form = 'D',field = 'C'):
    if field == 'R':
        Evec, Eval = _setR(*eig(mat,form))
    elif field == 'C':
        Evec, Eval = eig(mat,form)
    else:
        print('Field can only be C or R')
        return 'Error';
    U = np.concatenate(list(Evec.values()),axis=1)
    Mat = la.inv(U).dot(mat).dot(U)
    return U,_realize(Mat)   


