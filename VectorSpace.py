import sympy as sym
import numpy as np
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

def Exp(poly,op):
    coef = poly.all_coeffs()
    exp = [c*op**i for (c,i) in zip(coef,range(len(coef)-1,0,-1))]
    exp.append(coef[-1]*sym.eye(op.shape[0]))
    return sum(exp,sym.zeros(op.shape[0]))

def gen_eigenvects(mat):
    E = mat.eigenvals()
    x = sym.Symbol('x')
    gen_eigvec = {}
    gen_eigval = {}
    for egvl in E.keys():
        geo_mlt = len(Exp(sym.Poly(x-egvl,x),mat).nullspace())
        for i in range(E[egvl]):
            gen_eigvec[egvl] = Exp(sym.Poly(x-egvl,x)**(i+1),mat).nullspace()
            if len(gen_eigvec[egvl]) == E[egvl]:
                gen_eigval[egvl] = (E[egvl],geo_mlt,i + 1)
                break
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    return flatten(list(gen_eigvec.values())),gen_eigval