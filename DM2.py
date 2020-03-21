##
# Algos fait par :
#    Alexis Direz
# et Neil Ségard

## Imports et fonctions de base:

import numpy as np
from copy import deepcopy

N = 4

X_first = np.array([1,7,3,4], float)
A = np.arange(N**2).reshape(N,N)
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

def G(X):
    n = len(X)
    # print(n)
    tab = np.zeros(n)
    for i in range(n):
        tab[i] = X[i] + X[i-1]*(-1)**i  #je chois les contraintes valant xi plus ou moins la valeur du xi precedent
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
    #2 mettre sous la forme CX = 0 les contraintes C2
    #3 Programmer uzawa pour les contraintes d'inégalité
    #4 Programmer uzawa pour les contraintes d'égalité
    #5 Programmer uzawa pour les contraintes mixtes
def Uzawa_simple():
    return 1



## Tests
# print(gradientDescenteProjete())
# print(Uzawa())