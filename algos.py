def f(x):
    return 2*x*x +x

def fp(x):
    return 4*x +1


def Newton(x, e):
    xold=x
    x=xold-f(xold)/fp(xold)

    while abs(xold-x)>e:
        print(x," ",xold)
        xold=x
        x=xold-f(xold)/fp(xold)

    return x



print(Newton(1, 0.001))



