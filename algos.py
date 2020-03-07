import numpy as np #seulement pour : decomposition LU

def g(x):
    return np.square(x)

def gp(x):
    return np.multiply(2,x)

def gradfa(X):
    global compteur
    compteur=compteur+1
    d=(1+a*X[0]**2+X[1]**2)**2
    return 2/d*array([a,1])*X

## Algo de descente  PAS FINI
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

## Algo du gradient à pas fixe  PAS FINI

def gradientPasFixe(f, fp, x, eCible, alpha):
    e=2*eCible

    while e>eCible:
        #print(e)
        w = 1#A FAIRE
        x = x + alpha * w
        e = abs(alpha * w)

    return x

## Algo du gradient à pas optimal  PAS FINI
def gradientPasFixe(f, fp, x, eCible):
    e=2*eCible

    while e>eCible:
        #print(e)
        w = 1#A FAIRE
        alpha=1#A FAIRE
        x = x + alpha * w
        e = abs(alpha * w)

    return x

## Méthode de section dorée MARCHE PAS
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

## Algo de Newton
def Newton(f, fp, x, e):
    xold=x
    x=xold-f(xold)/fp(xold)

    while abs(xold-x)>e:
        #print(x," ",xold)
        xold=x
        x=xold-f(xold)/fp(xold)

    return x

## Décomposition LU
def res_sys_line_diago(A_diago,b):
    I= np.shape(A)[0]
    X=np.zeros((I,1))
    for i in range(I):
        X[i]=b[i]/A_diago[i][i]
    return X

def res_sys_line_triang_inf(A,b):
    I,N = np.shape(A)
    X=np.zeros((I,1))
    X[0]=b[0]/A[0][0]
    for i in range(1,I):
        sum=0
        for k in range(0,i-1):
             sum+=A[i][k]*X[k]
        X[i]=(1/A[i][i])*(b[i]-sum)
    return X


def res_sys_line_triang_sup(A,b):
    I = np.shape(A)[0]
    X=np.zeros((I,1))
    X[I-1]=b[I-1]/A[I-1][I-1]
    for i in range(I-2,-1,-1):
        sum=0
        for k in range(i,I):
            sum+=A[i][k]*X[k]
        #print("sum= ",sum)
        X[i]=(1/A[i][i])*(b[i]-sum)
        #print("X[i]= ",X[i])
    return X


def decomposition_LU(A):
    n=np.shape(A)[0]
    L,U=np.zeros((n,n)) , np.zeros((n,n))
    for i in range(n):
        U[0][i]=A[0][i]
        L[i][0]=(A[i][1])/(U[0][0])
    for i in range(1,n):
        L[i][i]=1
        s=0
        for k in range(i-1):
            s+=L[i][k]*U[k][i]
        U[i][i]=A[i][i]-s
        for j in range(i,n):
            s1=0
            s2=0
            for k in range(i-1):
                s1+=L[i][k]*U[k][j]
                s2+=L[j][k]*U[k][i]
            U[i][j] = A[i][j]-s1
            L[j][i] = ( A[j][i] - s2 )/(U[i][i])

    return L,U

##tests
print("Test Descente")

print("\nTest gradient à pas fixe")

print("\nTest gradient à pas optimal")

print("\nTest Section dorée")
print(sectionDoree(-1,2,f,0.001))

print("\nTest Newton")
print(Newton(g, gp, 2, 0.001))
print(Newton(g, gp, -1, 0.001))

A=np.identity(4)
b=[1,1,1,2]
print(res_sys_line_triang_sup(A,b))
print(res_sys_line_diago(A,b))