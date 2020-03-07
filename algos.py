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

        d=1#A FAIRE
        a=1#A FAIRE

        xold=x
        x=xold + a*d

    return x

##Algo du gradient à pas fixe  PAS FINI
def gradientPasFixe(f, fp, x, eCible, alpha):
    e=2*eCible

    while e>eCible:
        #print(e)
        w = 1#A FAIRE
        x = x + alpha * w
        e = abs(alpha * w)

    return x

##Algo du gradient à pas optimal  PAS FINI
def gradientPasFixe(f, fp, x, eCible):
    e=2*eCible

    while e>eCible:
        #print(e)
        w = 1#A FAIRE
        alpha=1#A FAIRE
        x = x + alpha * w
        e = abs(alpha * w)

    return x

##Méthode de section dorée MARCHE PAS
def sectionDoree(a,b,f,e):
    p=1.618
    xm = p*a + (1-p)*b
    xp = a+b-xm
    vm=f(xm)
    vp=f(xp)

    while b-a>=e:
        print("")
        print(a," ",b)
        print(xm," ",xp)
        print(vm," ",vp)
        if (vm<=vp):
            b=xp
            xp=xm
            xm=a+b-xp
            vp=vm
            vm=f(xm)
        else:
            a=xm
            xm=xp
            xp=a+b-xm
            vm=vp
            vp=f(xp)

    return (a+b)/2

##Algo de Newton
def Newton(f, fp, x, e):
    xold=x
    x=xold-f(xold)/fp(xold)

    while abs(xold-x)>e:
        #print(x," ",xold)
        xold=x
        x=xold-f(xold)/fp(xold)

    return x

##tests
print("Test Descente")

print("\nTest gradient à pas fixe")

print("\nTest gradient à pas optimal")

print("\nTest Section dorée")
print(sectionDoree(-1,2,f,0.001))

print("\nTest Newton")
print(Newton(g, gp, 2, 0.001))
print(Newton(g, gp, -1, 0.001))