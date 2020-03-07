import numpy as np

def g(x):
    return np.square(x)

def gp(x):
    return np.multiply(2,x)




##Algo de descente  PAS FINI
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

##Algo de Newton
def Newton(f, fp, x, e):
    xold=x
    x=xold-f(xold)/fp(xold)

    while abs(xold-x)>e:
        print(x," ",xold)
        xold=x
        x=xold-f(xold)/fp(xold)

    return x

##Algo du gradient descente à pas fixe


##Algo du gradient descente à pas optimal


##Méthode de section dorée


##tests
print(Newton(g, gp, 2, 0.001))
print(Newton(g, gp, -1, 0.001))