"""
Credit : Ramin Abbasi
Update : 02/18/2021
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
    1) prod : to multiply two linMaps
    2) null : returns the null of a linMap
    3) eig  : returns the eigenval, eigenvec of linMap
    4) Adj : returns the adjoint of linMap
    
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
Gramm(base) - generate an orthonormal basis from 'base' using Gram-Shmidt
Matv(vec,base) - returns the matrix of 'vec' using 'base'
InvMatv(mat,base) - returns the vector of 'mat' using 'base'
Mat(lMap,vBase,wBase) - returns the matrix of 'lMap' using two bases 'vBase','wBase'
invMat(mat,vBase,wBase) - returns the linMap of 'mat' using two bases 'vBase','wBase'
getD(space) - returns the dimension of 'space'
U(vBase,wBase) - returns the Unitary transformation from 'vBase' to 'wBase'
isindep(bList) - checks the linear independence of 'bList'
sym2num(Mat) - turns the sympy matrix into numpy array
"""



import sympy as sym
import numpy as np
#import scipy.linalg as la
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
        M = Mat(self,basis(self.V),basis(self.W)).eigenvals()
        print(M)
        return 0
    def Adj(self):
        vBase = basis(self.V)
        wBase = basis(self.W)
        return invMat(sym.Matrix.adjoint(Mat(self,vBase,wBase)),wBase,vBase)

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
    I = invMat(sym.eye(n),basis(space),basis(space))
    return I

def basis(space, gramm = 1):
    n = getD(space)
    if re.match(r'F.',space):
        base = sym.Matrix.diag([1]*n)
        B = vectors([base.row(i)[:] for i in range(n)],space)
        if gramm == 1: return Gramm(B)
        if gramm == 0: return B
    if re.match(r'P.',space):
        x = sym.Symbol('x')
        B = vectors([x**i for i in range(n)],space)
        if gramm == 1: return Gramm(B)
        if gramm == 0: return B

def Gramm(base):
    eBase = base.copy()
    eBase[0] = base[0].normalize()
    for i in range(1,len(base)):
        eBase[i] = (base[i]- sum([base[i].innerproduct(eBase[j])*eBase[j]\
                    for j in range(i)],base[0].initial())).normalize()
    return eBase

def Matv(v,base):
    if re.match(r'P.',base[0].space):
        x = sym.Symbol('x')
        base_sum = sum([b.vec for b in base])
        scale = np.array(sym.Poly(base_sum,x).all_coeffs())
        coef = np.array(sym.Poly(v.vec,x).all_coeffs())
        coef_pad = np.zeros(scale.shape)
        coef_pad[scale.shape[0]-coef.shape[0]:] = coef
        mat = coef_pad/scale
        return sym2num(mat[scale !=0][::-1])[:,np.newaxis]
    if re.match(r'F.',base[0].space):
        B = sym.Matrix([(b.vec).T for b in base]).T
        return sym2num(B.inv()*(v.vec)[:,np.newaxis])
    
def invMatv(mat,base):
    if mat != []:
        return vector(sum([mat[i]*base[i].vec for i in range(len(base))]),base[0].space)
    else:
        return zerov(base[0].space)

def Mat(lMap,vBase=0,wBase=0):
    if vBase == 0 : vBase = basis(lMap.V)
    if wBase == 0 : wBase = basis(lMap.W)
    return np.concatenate([Matv(lMap(VBase),wBase) for VBase in vBase],axis=1)

def invMat(mat,vBase=0,wBase=0):
    if vBase == 0 : vBase = basis("F{}".format(mat.shape[1]))
    if wBase == 0 : wBase = basis("F{}".format(mat.shape[0]))
    def F(x):
        ''' Unknown '''
        vx = vector(x,vBase[0].space)
        x_vec = Matv(vx,vBase)
        return (invMatv(mat*x_vec,wBase)).vec
    return linMap(F,vBase[0].space,wBase[0].space)
        
    
def getD(space):
    if re.match(r'F.',space):
        return int(re.search(r'\d',space).group())
    if re.match(r'P.',space):
        return int(re.search(r'\d',space).group())+1

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

