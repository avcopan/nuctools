import numpy
from nuctools.surfaces.cerjanmiller import f, g, h

p = (0.05, 0.1)

print(f((1, 0)))
print(g((1, 0)))
print(h((1, 0)))


from nuctools.rfo import rational_function_optimization

x = rational_function_optimization(f=f, x0=p, g=g, h=h, smax=1., order=1)
print(x)
