"""
Credit : Ramin Abbasi
Update : 03/16/2021
========================
Vector Space (VS) Package
========================
This package includes tools from linear algebra to work on linear maps and
matrices. It requires sympy, numpy, scipy, and re packages. 


Classes
=======
VS includes two main classes:
    1) linMap
    2) vector

linMap
------
It generates a linear map object (V -> W) with three attributes:
    1) fun: fun: sets the functionality of linear map
    2) V: str: sets the vector space V (e.g F3,P2)
    3) W: str: sets the vector space W (e.g F5,P1)
and the following methods:
    1) doc  : sets the linMap documentation
    2) prod : multiplies two linMaps
    3) null : nullspace of a linMap
    4) eig  : eigen_val and eigen_vec of linMap
    5) trace: trace of the operator
    6) det  : determinant of the operator
    7) char : characteristic polynomial of the operator
    8) diag : triangular or diagonal form of the operator
    9) Adj  : adjoint of linMap
    10) svd : singular value decomposition
    
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
    5) project : project a vector to the span of the orthonormal base 'e_basis'

Functions on Linear Maps
========================

vectors(array,space) - generate a list of vectors from 'array' in 'space'
zerov(space) - generates a zero vector in 'space'
eye(space) - generate identity map on 'space'
basis(space) - generate orthonormal standard basis on 'space' 
Gram(base) - generate an orthonormal basis from 'base' using Gram-Shmidt
Matv(vec,base) - returns the matrix of 'vec' using 'base'
InvMatv(mat,base) - returns the vector of 'mat' using 'base'
Mat(lMap,vBase,wBase) - returns the matrix of 'lMap' using two bases 'vBase','wBase'
invMat(mat,vBase,wBase) - returns the linMap of 'mat' using two bases 'vBase','wBase'
isindep(l) - Checks if a list of vectors is linearly independent
getD(space) - returns the dimension of 'space'
U(vBase,wBase) - returns the Unitary transformation from 'vBase' to 'wBase'
sym2num(Mat) - turns the sympy matrix into numpy array
realize(Mat) - sets small floats to zero
"""
import warnings 
import sympy as sym
import numpy as np
import scipy.linalg as la
import re

#========================================================================= 
# Classes
#========================================================================= 

class linMap:
    def __init__(self,fun,V,W):
        self.fun = fun
        self.V = V
        self.W = W
    def __call__(self,v):
        if re.match(r'P.',self.V): return sym2num(vector(self.fun(v.vec).evalf(6),self.W))
        if re.match(r'F.',self.V): return sym2num(vector(self.fun(v.vec),self.W))
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
            return 'lMap({}->{}): {} ==> {} {}'.format(self.V,self.W,z, self.fun(z),self.fun.__doc__)
        if re.match(r'P.',self.V):
            n = getD(self.V)
            p = invMatv(np.ones(n),basis(self.V))
            return 'lMap({}->{}): {} ==> {} {}'.format(self.V,self.W,p,self(p),self.fun.__doc__)
    def __repr__(self):
        return str(self)
    def doc(self,document):
        self.fun.__doc__ = document
    def prod(self,other):
        Map = linMap(lambda x: self.fun(other.fun(x)),other.V,self.W)
        Map.fun.__doc__ = ""
        return Map
    def null(self):
        vBase = Gram(basis(self.V))
        wBase = Gram(basis(self.W))
        N = la.null_space(Mat(self,vBase,wBase)).transpose()
        if N.size == 0 : return [zerov(self.V)]
        return [invMatv(M,vBase) for M in N]
    def eig(self):
        vBase = Gram(basis(self.V))
        wBase = Gram(basis(self.V))
        n = getD(self.V)
        M = Mat(self,vBase,wBase)
        eigen_val, alg_mlt = np.unique(la.eigvals(M),return_counts=True)
        eigen_vec = []
        eig_mlt = []
        for ev,algmlt in zip(eigen_val,alg_mlt):
            nill = M - ev*np.eye(n)
            geo_mlt = la.null_space(nill,rcond=1e-10).shape[1]
            eigen_vec.append(la.null_space(np.linalg.matrix_power(nill,algmlt),rcond=1e-10))
            eig_mlt.append((ev,algmlt,geo_mlt))
        eigen_vec = np.concatenate(eigen_vec,axis=1)
        eigen =[invMatv(vec[:,np.newaxis],vBase) for vec in eigen_vec.transpose()]
        return eig_mlt,sym2num(eigen)
    def trace(self):
        return sum([ev[1]*ev[0] for ev in self.eig()[0]])
    def det(self):
        return np.prod(np.array([ev[0]**ev[1] for ev in self.eig()[0]]))
    def char(self):
        z = sym.symbols('z')
        poly = [(z-ev[0])**ev[1] for ev in self.eig()[0]]
        return sym.expand(np.prod(np.array(poly)))
    def diag(self):
        eig_mlt, eigen = self.eig()
        return Mat(self,eigen,eigen)
    def Adj(self):
        vBase = Gram(basis(self.V))
        wBase = Gram(basis(self.W))
        return invMat(Mat(self,vBase,wBase).transpose().conjugate(),wBase,vBase)
    def svd(self):
        sv_right = (self.Adj().prod(self)).eig() 
        sv_left = (self.prod(self.Adj())).eig()
        sv = [np.sqrt(ev[0]) for ev in sv_right[0]]
        e = sv_right[1]
        f = sv_left[1]
        return sv,e,f

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
            return sym.integrate(self.vec*other.vec,(sym.symbols('x'),-1,1))
    def norm(self):
        return sym.sqrt(self.innerproduct(self))
    def normalize(self):
        return self/self.norm()
    def initial(self):
        return vector(self.vec-self.vec,self.space)
    def project(self,e_base):
        return sum([self.innerproduct(b)*b for b in e_base],e_base[0].initial())

    
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
    I = invMat(sym.eye(n),basis(space),basis(space),doc = "Identity")
    return I

def basis(space):
    n = getD(space)
    if re.match(r'F.',space):
        base = sym.Matrix.diag([1]*n)
        B = vectors([base.row(i)[:] for i in range(n)],space)
        return B
    if re.match(r'P.',space):
        x = sym.symbols('x')
        B = vectors([x**i for i in range(n)],space)
        return B

def Gram(base):
    eBase = base.copy()
    eBase[0] = base[0].normalize()
    for i in range(1,len(base)):
        eBase[i] = (base[i]- sum([base[i].innerproduct(eBase[j])*eBase[j]\
                    for j in range(i)],base[0].initial())).normalize()
    return eBase

def Matv(v,base, symnum = True, indep_prompt = True):
    if re.match(r'P.',base[0].space):
        n = len(base)
        c = sym.symbols('c0:{}'.format(n))
        x = sym.symbols('x')
        C = np.array(c)[:,np.newaxis]
        B = np.array(base)

        Base_Poly = sym.Poly(str(invMatv(C,base)),x)
        Vec_Poly  = sym.Poly(v.vec,x)

        Eq = (Base_Poly-Vec_Poly).all_coeffs()
        Coef = sym.linsolve(Eq,c)
        if len(Coef) < 1 or len(Coef.free_symbols) > 0:
            if indep_prompt: 
                print('Warning: Improper base! Returning 0')
            return 0
        res = np.array(Coef.args[0])[:,np.newaxis]
        return res.astype('float') if symnum else res

    if re.match(r'F.',base[0].space):
        B = sym.Matrix([(b.vec).T for b in base]).T
        res = B.pinv()*(v.vec)[:,np.newaxis]
        chk = B*B.pinv()*v.vec[:,np.newaxis]-v.vec[:,np.newaxis]
        if B.rank() < B.shape[1] or chk.norm() > 1e-6:
            if indep_prompt:
                print('Warning: Improper base! Returning 0')
            return 0
        return sym2num(res) if symnum else res

def invMatv(mat,base):
    vec = sum([mat[i]*base[i].vec for i in range(len(base))])
    if re.match(r'F.',base[0].space): return vector(vec,base[0].space)
    if re.match(r'P.',base[0].space): return vector(sym.Matrix([vec])[0],base[0].space)

def Mat(lMap,vBase,wBase):
    return np.concatenate([Matv(lMap(VBase),wBase) for VBase in vBase],axis=1)

def invMat(mat,vBase,wBase, doc = ""):
    def F(x):
        F.__doc__ = doc
        vx = vector(x,vBase[0].space)
        x_vec = Matv(vx,vBase,symnum=False)
        if isinstance(mat,(np.ndarray)) and isinstance(x_vec,(np.ndarray)):
            x_vec = sym.Matrix(x_vec)
        return (invMatv(mat*x_vec,wBase)).vec
    return linMap(F,vBase[0].space,wBase[0].space)

def isindep(l):
    for j in range(1,len(l)):
        if not isinstance(Matv(l[j],l[:j],indep_prompt=False),(int)): 
            return False
    return True
def mkindep(l):
    pass

def getD(space):
    if re.match(r'F.',space):
        return int(re.search(r'\d',space).group())
    if re.match(r'P.',space):
        return int(re.search(r'\d',space).group())+1

def U(vBase,wBase):
    return Mat(eye(vBase[0].space),vBase,wBase)

def sym2num(Mat):
    if isinstance(Mat,(list)):
        return [sym2num(v) for v in Mat]
    if isinstance(Mat,(vector)):
        if isinstance (Mat.vec,(np.ndarray)):
            return vector((Mat.vec).astype('float'),Mat.space)
        if isinstance (Mat.vec,(sym.Basic)):
            return vector((Mat.vec).evalf(),Mat.space)
    return np.array(Mat).astype('float')

def realize(Mat,tol = 1e-10):
    if isinstance (Mat,(list)):
        return [realize(v) for v in Mat]
    if isinstance (Mat,(vector)):
        return realize(Mat.vec)
    M = Mat.astype(np.complex)
    M.real[abs(Mat.real) < tol] = 0.0
    M.imag[abs(Mat.imag) < tol] = 0.0
    if (abs(Mat.imag) < tol).all():
        return np.real(M)
    return M


