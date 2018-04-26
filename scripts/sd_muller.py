import numpy


def energy(z):
    x, y = z
    x0 = numpy.array((1, 0, -0.5, -1))
    y0 = numpy.array((0, 0.5, 1.5, 1))
    e = numpy.array((-200, -100, -170, 15))
    a = numpy.array((-1, -1, -6.5, 0.7))
    b = numpy.array((0, 0, 11, 0.6))
    c = numpy.array((-10, -10, -6.5, 0.7))
    return numpy.sum(e * numpy.exp(
        a * (x - x0) ** 2 + b * (x - x0) * (y - y0) + c * (y - y0) ** 2))


# minima:
print(energy((-0.558, +1.442)))
print(energy((+0.623, +0.028)))
print(energy((-0.050, +0.467)))
