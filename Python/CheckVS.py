"""
Checks VectorSpace functionality by touching almost all defined functions
"""
import VectorSpace as vs
import sympy as sym
import numpy as np

x = sym.Symbol('x')

vBase = vs.vectors([[1,2,0],[0,1,3],[1,0,1]],'F3')
wBase = vs.vectors([[1,0,0],[0,1,0],[0,0,1]],'F3')

pBase = vs.vectors([1,x,x**2-1],'P2')
qBase = vs.vectors([1,x],'P1')

vList = vs.vectors([[0,0,0],[1,1,0],[0,0,0],[2,1,0]],'F3')
pList = vs.vectors([0,1,x,0,x**2-1,2*x],'P2')

v = vs.vector([1,2,3],'F3')
p = vs.vector(1+2*x+3*x**2,'P2')


def f(X):
    """A Linear Map!"""
    return np.array([X[0]+X[1]+X[2],2*X[1],2*X[2]])
def g(p):
    """ Differential Map!"""
    return sym.diff(p,x)
def h(X):
    """A Linear Map!"""
    return np.array([X[0],X[0]+X[1],0])    

v0 = vs.zerov('F3')
p0 = vs.zerov('P1')

T = vs.linMap(f,'F3','F3')
R = vs.linMap(g,'P2','P1')
S = vs.linMap(h,'F3','F3')

w = T(v)
q = R(p)

v_vec = vs.Matv(v,wBase)
p_vec = vs.Matv(p,pBase)

T_mat = vs.Mat(T,vBase,wBase)
R_mat = vs.Mat(R,pBase,qBase)


v_vec_inv = vs.invMatv(v_vec,wBase)
p_vec_inv = vs.invMatv(p_vec,pBase)

V_vec = vs.Matv(v_vec_inv,wBase)
P_vec = vs.Matv(p_vec_inv,pBase)


T_mat_inv = vs.invMat(T_mat,vBase,wBase)
R_mat_inv = vs.invMat(R_mat,pBase,qBase)

T_null = T.null()
R_null = R.null()

T_range = T.range()
R_range = R.range()

svT,eT,fT = T.svd()
svR,eR,fR = R.svd()


print("Basis (F2): ",vs.basis('F2'))#
print("Basis (P1): ",vs.basis('P1'))#
print("vList: ", vList)#
print("pList: ", pList)#
print("Is vList indep: ", vs.isindep(vList))#
print("Is pList indep: ", vs.isindep(pList))#
print("vList made indep: ", vs.mkindep(vList))#
print("pList made indep: ", vs.mkindep(pList))#
print("vList made basis (F3): ", vs.mkbasis(vList,'F3'))#
print("pList made basis (P3): ", vs.mkbasis(pList,'P3'))#
print("v: ",v)#
print("p: ",p)#
print("v0: ",v0)#
print("p0: ",p0)#
print("I (F2): ",vs.eye('F2'))#
print("I (P1): ",vs.eye('P1'))#
print("T: ",T)#
print("R: ",R)#
print("T(v+w) == T(v) + T(w): ",T(v+w) == T(v) + T(w))#
print("R(p+q) == R(p) + R(q): ",T(v+w) == T(v) + T(w))#
print("T*(T+S) == T*T + T*S: ",T*(T+S) == T*T + T*S)#
print("T+T == 2*T:",T+T == 2*T)#
print("T(v): ",w)#
print("R(p): ",q)#
print("Null(T): ",T_null)#
print("Null(R): ",R_null)#
print("Range(T): ",T_range)#
print("Range(R): ",vs.realize(R_range))#
print("T.inv(): ", T.inv())#
print("T.pinv(): ", vs.realize(T.pinv()))#
print("is T injective: ", vs.isinj(T))#
print("is R injective: ", vs.isinj(R))#
print("is T surjective: ", vs.issurj(T))#
print("is R surjective: ", vs.issurj(R))#
print("eigen(T): ",T.eig()[0])#
print("trace(T): ",T.trace())#
print("det(T)  : ",T.det())#
print("Char(T) : ",T.char())#
print("T*: ",T.adj())#
print("R*: ",vs.realize(R.adj()))#
print("M(v):\n",v_vec)#
print("M(p):\n",p_vec)#
print("M(T):\n",T_mat)#
print("M(R):\n",R_mat)#
print("iM(M(v)): ",v_vec_inv)#
print("iM(M(p)): ",p_vec_inv)#
print("M(iM(M(v))):\n",V_vec)#
print("M(iM(M(p))):\n",P_vec)#
print("iM(M(T)): ",T_mat_inv)#
print("iM(M(R)): ",R_mat_inv)#
print("iM(M(T))(v): ",vs.realize(T_mat_inv(v)))#
print("iM(M(R))(p): ",R_mat_inv(p))#


print("Diag(T) :\n",vs.realize(T.diag()))
print("M(T,e,f):\n",vs.realize(vs.Mat(T,eT,fT)))
print("M(R,e,f):\n",vs.realize(vs.Mat(R,eR,fR),digits = 3))
print("U:\n",vs.realize(vs.U(wBase,vBase),digits=3))