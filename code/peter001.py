from __future__ import print_function, division
import numpy as np
from theano import *
import theano.tensor as T


def sa(a):
    return '%s.%s' % (list(a.shape), a.dtype)

def sb(a):
    return '%s %s' % (a, sa(a))


x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
a = z.eval({x : 16.3, y : 12.1})
print('a=%s' % a)


x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
b = f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
print('b=%s' % b)

print('x=%s' % sb(x))
print('y=%s' % sb(y))
print('z=%s' % sb(z))

a = theano.tensor.vector()          # declare variable
b = theano.tensor.vector()
out =  a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression (a + b) ^ 2
f = theano.function([a, b], out)    # compile function
r1 = f([0, 1], [2, 3])
r2 = f([2, 2], [2, 3])
print(r1, sa(r1))     # prints `array([2, 4])`
print(r2, sa(r2))     # prints `array([16, 25])`


print('-' * 80)
print('fmatrix')
x = T.fmatrix()
print('x=%s' % sb(x))

x = T.scalar('myvar', dtype='float64')
print('x=%s' % sb(x))
x = T.iscalar('myvar')
print('x=%s' % sb(x))
x = T.TensorType(dtype='uint32', broadcastable=())('myvar')
print('x=%s' % sb(x))
