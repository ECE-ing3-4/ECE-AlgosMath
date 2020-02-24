a=-1
b=2
e=0.0001

p= 1.618

def f(x):
    return x*x

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

print((a+b)/2)