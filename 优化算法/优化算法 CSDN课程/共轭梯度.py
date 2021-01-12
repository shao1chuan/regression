import numpy as np
def conj_grad(A,b):
    alphas = []
    n=b.shape[0]

    xs=[]
    rs=[]
    ps=[]

    x0=np.random.rand(b.shape[0])
    xs.append(x0)

    r0=b-A.dot(x0)
    rs.append(r0)

    p0=r0
    ps.append(p0)

    alpha0=p0.dot(p0)/p0.dot(A).dot(p0)
    alphas.append(alpha0)

    for i in range(n):

        r=rs[i]-alphas[i]*A.dot(ps[i])
        rs.append(r)
        beta=r.dot(r)/(rs[i].dot(rs[i]))

        alpha=ps[i].dot(rs[i])/(ps[i].dot(A).dot(ps[i]))
        alphas.append(alpha)

        x=xs[i]+alpha*ps[i]
        xs.append(x)

        p=r+beta*ps[i]
        ps.append(p)
    return xs
A=np.array([[4,1],[1,3]])
b=np.array([1,2])
np.linalg.inv(A).dot(b)
print(conj_grad(A,b))