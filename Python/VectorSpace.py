"""
Credit : Ramin Abbasi
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
    4) rnge: range of a linMap
    5) eig  : eigen_val and eigen_vec of linMap
    6) trace: trace of the operator
    7) det  : determinant of the operator
    8) char : characteristic polynomial of the operator
    9) diag : triangular or diagonal form of the operator
    10) adj : adjoint of linMap
    11) inv : inverse of linMap
    12) pinv: pseudo-inverse of linMap
    13) svd : singular value decomposition
    
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
mkindep(l) - Reduces a list of vectors into a list of independent vectors
mkbasis(l) - Turns (reduce or extend) a list into a basis
getD(space) - returns the dimension of 'space'
U(vBase,wBase) - returns the Unitary transformation from 'vBase' to 'wBase'
real(Mat) - rounds small floats to zero
"""

import sympy as sym
import numpy as np
import scipy.linalg as la
import re
sym.init_printing(pretty_print=False)
tol = 1e-8
tol_int = int(str(tol)[-1])
#========================================================================= 
# Classes
#========================================================================= 

class linMap:
    def __init__(self,fun,V,W):
        self.fun = fun
        self.V = V
        self.W = W
    def __call__(self,v):
        if isinstance(v,(list)) : return [self(vi) for vi in v]
        return vector(self.fun(v.vec),self.W)
    def __mul__(self,other):
        if type(self) == type(other):
            return self.prod(other)
        else:
            return linMap(lambda x: other*self.fun(x),self.V,self.W)
    __rmul__=__mul__
    def __pow__(self,scalar):
        if scalar == 0:
            return eye(self.V)
        if scalar == 1:
            return self
        return self*(self**(scalar-1))
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
            return 'lMap({}->{}): {} ==> ({}) {}'.format(self.V,self.W,z,str(self.fun(z))[1:-1],self.fun.__doc__)
        if re.match(r'P.',self.V):
            n = getD(self.V)
            p = invMatv(np.ones(n),basis(self.V))
            return 'lMap({}->{}): {} ==> {} {}'.format(self.V,self.W,p,self(p),self.fun.__doc__)
    def __repr__(self):
        return str(self)
    def __eq__(self,other):
        if self.V == other.V and self.W == other.W:
            return self.rnge() == other.rnge()
        else: return False
    def __ne__(self,other):
        return not (self == other)
    def __neg__(self):
        return -1*self
    def doc(self,document):
        self.fun.__doc__ = document
    def riesze(self):
        base = Gram(basis(self.V))
        if re.match(r'F.',self.V):
            l = [self(v)[0].conjugate()*v for v in base]
        if re.match(r'P.',self.V):
            l = [self(v).vec*v for v in base]
        return sum(l,zerov(base[0].space))
    def prod(self,other):
        Map = linMap(lambda x: self.fun(other.fun(x)),other.V,self.W)
        Map.fun.__doc__ = ""
        return Map
    def null(self):
        return null(self)
    def rnge(self):
        return rnge(self)
    def eig(self):
        return eig(self)
    def trace(self):
        return trace(self)
    def det(self):
        return det(self)
    def char(self):
        return char(self)
    def poly(self):
        return poly(self)
    def diag(self):
        return diag(self)
    def jordan(self):
        return jordan(self)
    def adj(self):
        return adj(self)
    def inv(self):
        return inv(self)
    def pinv(self):
        return pinv(self)
    def svd(self):
        return svd(self)


class vector:
    def __init__(self,array,space, product = 'std'):
        if re.match(r'F.',space):
            self.vec = np.array(array)
        if re.match(r'P.',space):
            self.vec = array
        self.space = space
        self.product = product
    def __mul__(self,other):
        if type(self) == type(other):
            return self.innerproduct(other)
        else:
            return vector(self.vec*other,self.space)
    __rmul__=__mul__
    def __truediv__(self,scalar):
        return self*(1/scalar)
    def __add__(self,other):
        return vector(self.vec+other.vec,self.space)
    def __sub__(self,other):
        return vector(self.vec-other.vec,self.space)
    def __str__(self):
        if re.match(r'F.',self.space): return '({})'.format(str(self.vec)[1:-1])
        if re.match(r'P.',self.space):
            return '{}'.format(self.vec)
    def __repr__(self):
        return str(self)
    def __eq__(self,other):
        if self.space == other.space:
            return (np.array(self.vec) == np.array(other.vec)).all()
        else: return False
    def __ne__(self,other):
        return not (self == other)
    def __neg__(self):
        return -1*self
    def __getitem__(self,index):
        return self.vec[index]
    def __setitem__(self,index,data):
        self.vec[index] = data
    def innerproduct(self,other):
        if self.product == 'std':
            if re.match(r'F.',self.space):
                return (self.vec.transpose().conjugate()).dot(other.vec)
            if re.match(r'P.',self.space):
                return sym.integrate(self.vec*other.vec,(sym.symbols('x'),-np.pi,np.pi))
        return self.product(self.vec,other.vec)
    def norm(self):
        if re.match(r'F.',self.space): return np.sqrt(self.innerproduct(self))
        if re.match(r'P.',self.space): return sym.sqrt(self.innerproduct(self))
    def normalize(self):
        return self/self.norm()
    def initial(self):
        return vector(self.vec-self.vec,self.space)
    def project(self,e_base):
        return sum([self.innerproduct(b)*b for b in e_base],e_base[0].initial())

    
#========================================================================= 
# VectorSpace Functions
#========================================================================= 

def vectors(array,space):
    return list(map(lambda x: vector(x,space) , array))


def zerov(space):
    if re.match(r'F.',space):
        return vector([0]*getD(space),space)
    if re.match(r'P.',space):
        return vector(0,space) 

def eye(space):
    n = getD(space)
    I = invMat(np.eye(n),basis(space),basis(space),doc = "Identity")
    return I

def basis(space):
    n = getD(space)
    if re.match(r'F.',space):
        base = np.eye(n)
        B = vectors(base,space)
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

def dual(base):
    n = getD(base[0].space)
    if re.match(r'F.',base[0].space): scalar_base = basis('F1')
    if re.match(r'P.',base[0].space): scalar_base = basis('P0')
    I = np.eye(n)
    return [invMat(I[j,:],base,scalar_base) for j in range(n)]

def Matv(v,base, indep_prompt = True):
    if isinstance (v,(list)): return [Matv(i,base, indep_prompt = True) for i in v]
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
        dtype = 'float'
        if any(abs(sym.im(c))>tol for c in Coef.args[0]): dtype = 'complex'
        res = np.array(Coef.args[0])[:,np.newaxis].astype(dtype)

    if re.match(r'F.',base[0].space):

        
        B = np.array([(b.vec).transpose() for b in base]).transpose()
        res = np.array(la.pinv(B).dot(v.vec[:,np.newaxis]))
        chk = B.dot(res)-v.vec[:,np.newaxis]

        if res.dtype != 'object' and la.norm(chk) > tol:
            if indep_prompt:
                print('Warning: Improper base! Returning 0')
            return 0
    return res
        

def invMatv(mat,base):
    if isinstance (mat,(list)): return [invMatv(i,base) for i in mat]
    vec = sum([(mat[i]*base[i].vec) for i in range(len(base))])
    if re.match(r'F.',base[0].space): return vector(vec,base[0].space)
    if re.match(r'P.',base[0].space): return vector(sym.Matrix([vec])[0],base[0].space)

def Mat(lMap,vBase,wBase):
    if isinstance (lMap,(list)): return [Mat(i,vBase,wBase) for i in lMap]
    return np.concatenate([Matv(lMap(VBase),wBase) for VBase in vBase],axis=1)

def invMat(mat,vBase,wBase, doc = ""):
    if isinstance (mat,(list)): return [invMat(i,vBase,wBase, doc = "") for i in mat]
    def F(x):
        F.__doc__ = doc
        vx = vector(x,vBase[0].space)
        x_vec = Matv(vx,vBase)
        return (invMatv(mat.dot(x_vec),wBase)).vec
    return linMap(F,vBase[0].space,wBase[0].space)

def isindep(l):
    zero_idx = 0
    while l[zero_idx].norm() < tol: 
        zero_idx += 1
        if zero_idx == len(l)-1 or len(l) == 1: break
    for j in range(1,len(l)):
        if not isinstance(Matv(l[j],l[:j],indep_prompt=False),(int)): 
            return False
    return True and zero_idx == 0

def mkindep(l):
    zero_idx = 0
    while l[zero_idx].norm() < tol: 
        zero_idx += 1
        if zero_idx == len(l)-1 : break
        if len(l) == 1: return l
    l = l[zero_idx:]
    out = [l[0]]
    for v in l[1:]:
        if isindep([*out,v]): out.append(v)
    return out   

def mkbasis(l,space):
    extended = [*l,*basis(space)]
    return mkindep(extended)

def isinj(T):
    Null = mkindep(T.null())
    if len(Null) == 1 and Null[0].norm() < tol :
        return True
    else: return False
def issurj(T):
    Range = T.rnge()
    if len(Range) == getD(T.W) and Range[0].norm() > tol:
        return True
    else: return False
def isbij(T):
    return issurj(T) and isinj(T)

def getD(space):
    if re.match(r'F.',space):
        return int(''.join(re.findall(r'\d',space)))
    if re.match(r'P.',space):
        return int(''.join(re.findall(r'\d',space)))+1

def U(vBase,wBase):
    return Mat(eye(vBase[0].space),vBase,wBase)


#========================================================================= 
# Linear Map Methods/Functions
#========================================================================= 
def null(T):
    vBase = Gram(basis(T.V))
    wBase = Gram(basis(T.W))
    N = la.null_space(Mat(T,vBase,wBase)).transpose()
    if N.size == 0 : return [zerov(T.V)]
    return [invMatv(M,vBase) for M in N]
def rnge(T):
    vBase = basis(T.V)
    return mkindep([T(v) for v in vBase])
def eig(T):
    vBase = Gram(basis(T.V))
    wBase = Gram(basis(T.W))
    if len(vBase) != len(wBase):
        print('This is not an operator! Returning 0')
        return 0
    n = getD(T.V)
    M = Mat(T,vBase,wBase)
    eigen_val, alg_mlt = np.unique(la.eigvals(M),return_counts=True)
    eigen_vec = []
    eig_mlt = []
    for ev,algmlt in zip(eigen_val,alg_mlt):
        nill = M - ev*np.eye(n)
        geo_mlt = la.null_space(nill,rcond=tol).shape[1]
        eigen_vec.append(la.null_space(np.linalg.matrix_power(nill,algmlt),rcond=tol))
        eig_mlt.append((ev,algmlt,geo_mlt))
    eigen_vec = np.concatenate(eigen_vec,axis=1)
    eigen =[invMatv(vec[:,np.newaxis],vBase) for vec in eigen_vec.transpose()]
    return eig_mlt,eigen

def trace(T):
    return sum([ev[1]*ev[0] for ev in T.eig()[0]])
def det(T):
    return np.prod(np.array([ev[0]**ev[1] for ev in T.eig()[0]]))
def char(T):
    z = sym.symbols('z')
    poly = [(z-ev[0])**ev[1] for ev in T.eig()[0]]
    return sym.expand(np.prod(np.array(poly)))

def poly(T):
    z = sym.symbols('z')
    poly = [(z-ev[0])**ev[1] for ev in T.jordan()[2]]
    return sym.expand(np.prod(np.array(poly)))

def diag(T):
    eig_mlt, eigen = T.eig()
    return Mat(T,eigen,eigen)
def jordan(T):
    Base = []
    n = getD(T.V)
    I = eye(T.V)
    eigen = T.eig()[0]
    min_deg = []
    for e in eigen:
        N = T-e[0]*I
        Idx = []
        V = (N**n).null()
        for j in range(0,n+2):
            Idx.append(_find_idx(V,lambda x: not isindep([(N**j)(x)])))
        Index = [j-i for i,j in zip(Idx[:-1],Idx[1:])]
        min_deg.append((e[0],len(list(_find_idx(Index, lambda x: len(x) != 0)))))
        v_idx = sum([list(i) for i in Index],[])[::-1]
        v_list = [[V[i]] for i in v_idx]
        for v in v_list:
            for j in range(n+1):
                v.insert(0,N(v[0]))
        Base.append([mkindep(v) for v in v_list])
    Base = mkindep(sum(sum(Base,[]),[]))
    return Mat(T,Base,Base),Base,min_deg
def adj(T):
    vBase = Gram(basis(T.V))
    wBase = Gram(basis(T.W))
    return invMat(Mat(T,vBase,wBase).transpose().conjugate(),wBase,vBase)
def inv(T):
    if not isbij(T):
        print("Linear Map not invertible! Returning 0")
        return 0
    vBase = Gram(basis(T.V))
    wBase = Gram(basis(T.W))
    return invMat(la.inv(Mat(T,vBase,wBase)),wBase,vBase)    
def pinv(T):
    vBase = Gram(basis(T.V))
    wBase = Gram(basis(T.W))
    return invMat(la.pinv(Mat(T,vBase,wBase)),wBase,vBase)  
def svd(T):
    ev_right = (T.adj().prod(T)).eig()
    ev_left = (T.prod(T.adj())).eig()
    singular = sum([[ev]*ev[1] for ev in ev_right[0]],[])
    sv = [np.sqrt(ev[0]) for ev in singular]
    e = ev_right[1]
    f = ev_left[1]
    return sv,e,f
def polar(T):
    sv,e,f = T.svd()
    Scale = invMat(np.diag(sv),e,e)
    T_ef = np.around(Mat(T,e,f),tol_int)
    T_ef[T_ef > 0] =  1
    T_ef[T_ef < 0] = -1
    Isometry = invMat(T_ef,e,f)
    return Isometry,Scale
    

#========================================================================= 
# Utilities
#========================================================================= 
def real(Obj,digits = 3, Tol = tol):
    if isinstance (Obj,(list)):
        return [real(v,digits, Tol) for v in Obj]
    if isinstance(Obj,(linMap)):
        vBase = basis(Obj.V)
        wBase = basis(Obj.W)
        return invMat(real(Mat(Obj,vBase,wBase), digits, Tol),vBase,wBase)  
    if isinstance (Obj,(vector)):
        return vector(real(Obj.vec,digits,Tol),Obj.space)
    
    if isinstance(Obj,(sym.Basic)):
        if isinstance(Obj,(sym.Float)):
            return sym.Float(Obj,digits)
        else:
            return Obj.evalf(digits)
    if isinstance (Obj,(np.ndarray)):
        M = Obj.astype(np.complex)
        M = np.around(M,digits)
        M[M==0.] = 0.
        if (abs(M.imag) < Tol).all():
            return np.real(M)
        return M
#========================================================================= 
# Private Functions
#========================================================================= 
def _find_idx(lst, condition):
    return set([i for i, elem in enumerate(lst) if condition(elem)])
def _isF(space):
    pass
    