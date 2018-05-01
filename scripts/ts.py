import numpy
from nuctools.surfaces.mullerbrown import f, g, h

# minima
p1 = numpy.array([-0.5, +1.4])
p2 = numpy.array([+0.6, +0.0])
p3 = numpy.array([-0.1, +0.5])
# transition states
t12 = numpy.array([-0.8, 0.6])
t13 = numpy.array([+0.2, 0.3])


import matplotlib.pyplot as pyplot

def fcut_12(s):
    return f(p1[:, None]*(1.-s)[None, :] + p2[:, None]*s[None, :])

def gcut_12(s):
    return g(p1[:, None]*(1.-s)[None, :] + p2[:, None]*s[None, :])

def hcut_12(s):
    return h(p1[:, None]*(1.-s)[None, :] + p2[:, None]*s[None, :])


S = numpy.linspace(0, 1, num=1000)
print(numpy.shape(S))
print(numpy.shape(fcut_12(S)))
print(numpy.shape(gcut_12(S)))
print(numpy.shape(hcut_12(S)))
pyplot.plot(S, fcut_12(S))
# pyplot.plot(S, numpy.linalg.norm(gcut_12(S), axis=0))
# pyplot.plot(S, numpy.trace(hcut_12(S), axis1=0, axis2=1))
pyplot.show()

# guesses
s = 0.7
p12 = (1.-s)*p1 + s*t12
p21 = (1.-s)*p2 + s*t12
p13 = (1.-s)*p1 + s*t13
p31 = (1.-s)*p3 + s*t13


from nuctools.rfo import rational_function_optimization

x12 = rational_function_optimization(f=f, x0=p12, g=g, h=h, smax=0.1, order=1)
print(x12)
x21 = rational_function_optimization(f=f, x0=p21, g=g, h=h, smax=0.1, order=1)
print(x21)
x13 = rational_function_optimization(f=f, x0=p13, g=g, h=h, smax=0.1, order=1)
print(x13)
x31 = rational_function_optimization(f=f, x0=p31, g=g, h=h, smax=0.1, order=1)
print(x31)


# from nuctools.mm import mode_follow
# 
# x12 = mode_follow(f=f, x0=p12, g=g, h=h)
# print(x12)
# x21 = mode_follow(f=f, x0=p21, g=g, h=h)
# print(x21)
# x13 = mode_follow(f=f, x0=p13, g=g, h=h)
# print(x13)
# x31 = mode_follow(f=f, x0=p31, g=g, h=h)
# print(x31)
