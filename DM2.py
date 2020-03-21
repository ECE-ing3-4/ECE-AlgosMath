##
# Algos fait par :
#    Alexis Direz
# et Neil Ségard

## Imports et fonctions de base:

import numpy as np
from copy import deepcopy

N = 8
h=1/(N+1)

X_first = np.array([1,7,3,4,8,7,6,2], float)
A = 2 * np.eye(N).reshape(N,N)
for i in range(N):
    for j in range(N):
        if (j==i+1 or j==i-1):
            A[i,j]=-1
A= A / (h)**2
b = -8*np.ones(N).reshape(N,1)


def J(X):
    return 0.5 * np.vdot(np.dot(A,X),X) + np.vdot(b,X)
def grad_J(X):
    return (np.vdot(b , np.ones(np.shape(A)[0]) ) + (np.vdot(np.dot(A,X),X) / norm_eucl(X, np.zeros(np.shape(X)[0]))) )  # derivation de matrice

def prod_scal(X,Y):
    return np.vdot(X,Y)

def norme_deux(X,Y):
    n1= np.shape(X)
    n2= np.shape(Y)
    sum=0
    if (n1 !=n2):
        print("erreur dim")
        return 0
    else:
        for i in range(n1):
            sum+= (X[i]-Y[i])**2
    return np.sqrt(sum)

def norm_eucl(X,Y):
    return np.linalg.norm(X-Y)

def projection(X,plan):
    n = np.shape(plan)
    Zeros = np.zeros(n)
    norm = norm_eucl(plan,Zeros)

    return (prod_scal(X,plan)/norm)*plan

def G():
    tab = np.zeros(N)
    for i in range(N):
        t = (i+1)*h
        tab[i] = -1 + max(0, -10 * (t-0.4)**(2) + 0.625 )

    return tab


## Algo du gradient descente projeté
def gradientDescenteProjete(contraintes_g, precision, X0):
    X=deepcopy(X0)
    Y=deepcopy(X0+7)  #arbitraire de prendre 7
    pas = 0.05 #aussi arbitraire
    n1=np.shape(X)[0]
    n2 = np.shape(contraintes_g)[0]
    residu = norm_eucl(X,Y)
    Proj = np.zeros(np.shape(X0))
    compteur = 0
    while(residu > precision):
        X1 = deepcopy(X)
        X-= pas * grad_J(X)
        for i in range(n2):
            if (X[i] < contraintes_g[i]):
                Proj[i]=contraintes_g[i]
            else:
                Proj[i]=X[i]
        if(n2<n1):
            for i in range(n2,n1,1):
                Proj[i]=X[i]
        residu = norm_eucl(X,Proj)
        #contraintes_g = G(X)
        compteur+=1
    print(residu)
    print(precision)
    print(compteur)
    return X

## Algo d'Uzawa

#1 mettre sous la forme CX <= g les contraintes C1

C1 = -1 * np.eye(N)

def g(X):
    var = np.dot(C,X)
    for i in range(N):
        var-=G()[i]
    return var

#2 mettre sous la forme CX = 0 les contraintes C2

C2 = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if (i<3):
            if (j+i==4+i):
                C2[i,(j+i)]=1
                C2[i,(j+i+1)]=-1

def h(X):
    return np.dot(C2,X)

def Lagrangien(X,L1,Mu):
    return J(X) + prod_scal(L1,h(X))+prod_scal(Mu,g(X))

def Grad_x_Lagrangien(X,L1,Mu):
    return np.dot(A,X)-b + np.dot(np.transpose(C2),L1) + np.dot(np.transpose(C1),Mu)

def Grad_L1_Lagrangien(X,L1,Mu):
    return np.dot(np.transpose(X),np.transpose(C2))

def Grad_Mu_Lagrangien(X,L1,Mu):
    return np.transpose(np.dot(C1,X) - g(X))

    #3 Programmer uzawa pour les contraintes d'inégalité


    #4 Programmer uzawa pour les contraintes d'égalité
    #5 Programmer uzawa pour les contraintes mixtes
def Uzawa_simple():
    return 1



## Tests
# print(gradientDescenteProjete())
# print(Uzawa())