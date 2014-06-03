# With the following commandx, the parameter file can be generated
# (this will be much faster if done on a GPU)
# execfile("training.py")
# train_convnet()

# theano.config.exception_verbosity='high'
# layer1.output.eval({x: numpy.random.randn(784, 500)})

execfile("training.py")
batch_size=1
execfile("model.py")

f = file(params_file, 'rb')

for obj in [layer0, layer1, layer2, layer3]:
        load_layer(f, obj)

gradient = T.grad(layer3.p_y_given_x[0,0], x)

from scipy.optimize import minimize
import numpy
import pylab

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
    i = 1
    for C in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        res = calc_min(val, C)
        val = res.x
        pylab.subplot(l,6,6*(k-1)+i)
        i = i + 1
        draw_char(val)

pylab.axis('off')

for i in range(2, 7):
    global img
    img = valid_set_x[i:(i+1)].eval()[0]
    sol_path(5,i-1)

