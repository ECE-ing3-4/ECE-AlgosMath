import numpy as np

def g(x):
    return np.square(x)

def gp(x):
    return np.multiply(2,x)


def Newton(f, fp, x, e):
    xold=x
    x=xold-f(xold)/fp(xold)

    while abs(xold-x)>e:
        #print(x," ",xold)
        xold=x
        x=xold-f(xold)/fp(xold)

    return x

def descente(x):
    xold=x
    x=xold-f(xold)/fp(xold)

    while abs(xold-x)>e:
        print(x," ",xold)

        d=1
        a=1

        xold=x
        x=xold + a*d

    return x





print(Newton(g, gp, 1, 0.001))