##
# Algos fait par :
#    Alexis Direz
# et Neil Ségard

## Imports et fonctions de base:

import numpy as np

X = np.array([1,7,3,4], float)
A = np.arange(16).reshape(4,4)
b = np.array(np.random.random_sample(4)).reshape(4,1)
J = 0.5 * np.vdot(np.dot(A,X),X) + np.vdot(b,X)
grad_J = (np.vdot(b , np.ones(np.shape(A)[0]) ) + (np.vdot(np.dot(A,X),X) / norm_eucl(X, np.zeros(np.shape(X)[0]))) )  # derivation de matrice

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


## Algo du gradient descente projeté
def gradientDescenteProjete(J, contraintes_g, precision, X0):
    X=X0
    Y=X0 + 7 #arbitraire de prendre 7
    pas = 0.05 #aussi arbitraire
    residu = norm_eucl(X,Y)
    while(residu > precision):
        X0-= pas * 1
    return 1

## Algo d'Uzawa
    #1 mettre sous la forme CX<= g les contraintes c1
    #2 mettre sous la forme CX = 0 les contraintes C2
    #3 Programmer uzawa pour les contraintes d'inégalité
    #4 Programmer uzawa pour les contraintes d'égalité
    #5 Programmer uzawa pour les contraintes mixtes
def Uzawa_simple():
    return 1



## Tests
# print(gradientDescenteProjete())
# print(Uzawa())