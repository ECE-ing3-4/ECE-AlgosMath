##
# Algos fait par :
#    Alexis Direz
# et Neil Ségard

## Imports et fonctions de base:

import numpy as np
from copy import deepcopy

N = 8
h=1/(N+1)

X_first = np.random.randint(100,size=(N,1))
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

G = np.zeros(N) #creation des gi dans un vecteur
for i in range(N):
    t = (i+1)*h
    G[i] = -1 + max(0, -10 * (t-0.4)**(2) + 0.625 )



## Algo du gradient descente projeté

def gradientDescenteProjete(contraintes_g, precision, X0):

    X=deepcopy(X0)
    X1 = deepcopy(X)
    pas = 0.05 # arbitraire

    n1=np.shape(X)[0]
    n2 = np.shape(contraintes_g)[0]

    residu = 10 # initialisation

    Proj = np.zeros(np.shape(X0))
    compteur = 0
    while(residu > precision):

        X1= X1 - (pas * grad_J(X) ) * X1
        for i in range(n2):
            if (X1[i] < contraintes_g[i]):
                Proj[i]=contraintes_g[i]
            else:
                Proj[i]=X1[i]
        if(n2<n1):
            for i in range(n2,n1,1):
                Proj[i]=X1[i]
        residu = norm_eucl(X1,Proj)
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
        var-=G[i]
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
def Uzawa_contrainte_ineg(X0, tolerance):
    X=deepcopy(X0)
    X1=deepcopy(X)

    pas = 0.05

    Mu=np.random.uniform(0,100,N)
    Mu1 = Mu=np.random.uniform(0,100,N)

    test_x = 1
    test_mu = 1
    compteur = 0

    while((test_x>tolerance or test_mu>tolerance) and compteur < 500): # pas plus de 500 itérations on pourra augmanter au besoin
        Mu = Mu1
        X = X1
        # print("lool")

        Mu1 += pas * (np.dot(C1,X) - g(X) )
        for i in range(N): #cherche le max entre 0 et les mu_i
            Mu1[i]=max(0,Mu1[i])

        X1 = np.dot(np.linalg.inv(A) ,b - np.dot(np.transpose(C1),Mu) )

        test_x = norm_eucl(X1,X)
        test_mu = norm_eucl(Mu,Mu1)
        compteur+=1
    print(compteur)
    return X1,Mu1

    #4 Programmer uzawa pour les contraintes d'égalité

def Uzawa_contrainte_eg(X0, tolerance):
    X=deepcopy(X0)
    X1=deepcopy(X)
    pas = 0.05
    L=np.random.uniform(0,10,N)
    L1 = Mu=np.random.uniform(0,10,N)
    test_x = 1
    test_l = 1
    compteur = 0
    while((test_x>tolerance or test_l>tolerance) and compteur < 500): # pas plus de 500 itérations on pourra augmanter au besoin
        L = L1
        X = X1

        L1 += pas * (h(X))

        X1 = np.dot(np.linalg.inv(A) ,b - np.dot(np.transpose(C2),L) )

        test_x = norm_eucl(X1,X)
        test_l = norm_eucl(L,L1)

        compteur+=1
    print(compteur)
    return X1,L1

    #5 Programmer uzawa pour les contraintes mixtes

def Uzawa_contrainte_mixtes(X0, tolerance):
    X=deepcopy(X0)
    X1=deepcopy(X)

    Mu=np.random.uniform(0,10,N)
    Mu1 = Mu=np.random.uniform(0,10,N)

    pas = 0.05

    L=np.random.uniform(0,10,N)
    L1 = Mu=np.random.uniform(0,10,N)

    test_mu = 1
    test_x = 1
    test_l = 1

    compteur = 0
    while((test_x>tolerance or test_mu>tolerance) and compteur < 500): # pas plus de 500 itérations on pourra augmanter au besoin
        L = L1
        X = X1
        Mu1 = Mu

        L1 += pas * h(X) # = pas * (np.dot(C2,X))

        Mu1 += pas * g(X) # = pas * np.dot(C1,X)
        for i in range(N): #cherche le max entre 0 et les mu_i
            Mu1[i]=max(0,Mu1[i])

        X1 = np.dot(np.linalg.inv(A) ,b - np.dot(np.transpose(C2),L) - np.dot(np.transpose(C1),Mu) )

        test_x = norm_eucl(X1,X)
        test_l = norm_eucl(L,L1)
        test_mu = norm_eucl(Mu,Mu1)

        compteur+=1
    print(compteur)
    return X1,L1



## Tests
gradientDescenteProjete(g(X_first),0.01,X_first)

# Uzawa_contrainte_ineg(X_first,0.01)
# Uzawa_contrainte_eg(X_first,0.01)
# Uzawa_contrainte_mixtes(X_first,0.01)