import cPickle
import gzip
import os
import sys
import time

import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer
import convolutional_mlp
import numpy

def store_layer(f, layer):
    W = layer.W.get_value()
    b = layer.b.get_value()
    cPickle.dump(W, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(b, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_layer(f, layer):
    W = cPickle.load(f)
    b = cPickle.load(f)
    layer.W.set_value(W)
    layer.b.set_value(b)

learning_rate=0.1
n_epochs=200
dataset='mnist.pkl.gz'
nkerns=[20, 50]
batch_size=500

params_file='params.save'

rng = numpy.random.RandomState(42)

datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_valid_batches /= batch_size
n_test_batches /= batch_size

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

ishape = (28, 28)  # this is the size of MNIST images

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

execfile('model.py')

###############
# TRAIN MODEL #
###############

def train_convnet():
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    
    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
    
            iter = (epoch - 1) * n_train_batches + minibatch_index
    
            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
    
            if (iter + 1) % validation_frequency == 0:
    
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))
    
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
    
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
    
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
    
                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
    
            if patience <= iter:
                done_looping = True
                break
            
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    
    f = file(params_file, 'wb')

    for obj in [layer0, layer1, layer2, layer3]:
        store_layer(f, obj)
