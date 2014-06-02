# Train the model
execfile("training.py")

# With the following command, the parameter file can be generated
# (this will be much faster if done on a GPU)
#  train_convnet()

f = file(params_file, 'rb')

for obj in [layer0, layer1, layer2, layer3]:
        load_layer(f, obj)

gradient = T.grad(layer3.p_y_given_x[0,0], x)

from scipy.optimize import minimize
import numpy
import pylab

img = valid_set_x[2:3].eval()[0]
img = valid_set_x[4:5].eval()[0]

def draw_char(img):
    pylab.imshow(numpy.reshape(img, (28, 28)),
                 cmap=pylab.get_cmap("binary"), interpolation='none')

def show_char(img):
    draw_char(img)
    pylab.show()

def func(X, C=0.0):
    """ Objective function """
    return -float(layer3.p_y_given_x[0,0].eval({x: numpy.reshape(X, (784, 1))})) + C*numpy.linalg.norm(X - img)**2

# func(numpy.random.randn(784))

def func_deriv(X, C=0.0):
    """ Derivative of objective function """
    return -gradient.eval({x: numpy.reshape(X, (784, 1))})[:,0] + 2*C*(X - img)

# func_deriv(numpy.random.randn(784))

def calc_min(init, C):
    return minimize(func, init, args = (C,), jac=func_deriv,
                    method='L-BFGS-B', options={'disp': True, 'maxiter': 50},
                    tol=5e-13,
                    bounds = 784*[(0, 1)])

def sol_path(l=3,k=1):
    val = numpy.random.randn(784)
    i = 0
    for C in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        res = calc_min(val, C)
        val = res.x
        pylab.subplot(l,5,5*(k-1)+i)
        i = i + 1
        draw_char(val)



