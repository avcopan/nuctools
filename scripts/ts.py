import nuctools
from nuctools.muller import f, g, h

p1 = (-0.5, +1.4)
p2 = (+0.6, +0.0)
p3 = (-0.1, +0.5)

x1 = nuctools.cerjan_miller(f=f, x0=p1, g=g, h=h, order=1)
x2 = nuctools.cerjan_miller(f=f, x0=p2, g=g, h=h, order=1)
x3 = nuctools.cerjan_miller(f=f, x0=p3, g=g, h=h, order=1)

print(x1)
print(x2)
print(x3)
