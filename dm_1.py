## Liste des algos du cours
# - gradient descente a pas fixe
# - gradient descente à pas variable
# - resolution de systeme linéaire (diago, triang sup et inf)
# - Decomposition LU
# - Decomposition Cholesky

## Imports

import numpy as np

## Variables

A=np.identity(4)
b=[1,1,1,2]

## Algorithmes

def gradient_desc_a_pas_fixe(f,X,epsilon,alpha,nombre_max):
    res = X
    n=0
    erreur = 2*epsilon
    while (erreur> epsilon and n<=nombre_max):
        W =0 # a finir


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




