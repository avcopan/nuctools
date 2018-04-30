import nuctools
from nuctools.muller import f, g, h

# transition states
p4 = (-0.8, 0.6)
p5 = (+0.2, 0.3)

x4 = nuctools.cerjan_miller(f=f, x0=p4, g=g, h=h, order=1)
print(x4)
x5 = nuctools.cerjan_miller(f=f, x0=p5, g=g, h=h, order=1)
print(x5)
