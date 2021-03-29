"""
Checks VectorSpace functionality by touching almost all defined functions
"""
import VectorSpace as vs
import sympy as sym

x = sym.Symbol('x')

vBase = vs.vectors([[1,2,0],[0,1,3],[1,0,1]],'F3')
wBase = vs.vectors([[1,0,0],[0,1,0],[0,0,1]],'F3')

pBase = vs.vectors([1,x,x**2-1],'P2')
qBase = vs.vectors([1,x],'P1')

v = vs.vector([1,2,3],'F3')
p = vs.vector(1+2*x+3*x**2,'P2')

def f(X):
    """A Linear Map!"""
    return [X[0]+2*X[1]+X[2],2*X[1],3*X[2]]
def g(p):
    """ Differential Map!"""
    return sym.diff(p,x)

v0 = vs.zerov('F3')
p0 = vs.zerov('P1')

T = vs.linMap(f,'F3','F3')
R = vs.linMap(g,'P2','P1')


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

eigen_val,eigen_vec = T.eig()
svT,eT,fT = T.svd()
svR,eR,fR = R.svd()


print("Basis (F2): ",vs.basis('F2'))
print("Basis (P1): ",vs.basis('P1'))
print("v: ",v)
print("p: ",p)
print("v0: ",v0)
print("p0: ",p0)
print("I (F2): ",vs.eye('F2'))
print("I (P1): ",vs.eye('P1'))
print("T: ",T)
print("R: ",R)
print("T(v): ",w)
print("R(p): ",q)
print("M(v):\n",v_vec)
print("M(p):\n",p_vec)
print("M(T):\n",T_mat)
print("M(R):\n",R_mat)
print("iM(M(v)): ",v_vec_inv)
print("iM(M(p)): ",p_vec_inv)
print("M(iM(M(v))):\n",V_vec)
print("M(iM(M(p))):\n",P_vec)
print("iM(M(T)): ",T_mat_inv)
print("iM(M(R)): ",R_mat_inv)
print("iM(M(T))(v): ",T_mat_inv(v))
print("iM(M(R))(p): ",R_mat_inv(p))
print("T*: ",T.Adj())
print("R*: ",R.Adj())
print("eigen(T): ",T.eig()[0])
print("trace(T): ",T.trace())
print("det(T)  : ",T.det())
print("Char(T) : ",T.char())
print("Diag(T) :\n",vs.realize(T.diag(),tol=1e-10))
print("M(T,e,f):\n",vs.realize(vs.Mat(T,eT,fT),tol=1e-10))
print("M(R,e,f):\n",vs.realize(vs.Mat(R,eR,fR),tol=1e-10))
print("U:\n",vs.U(wBase,vBase))