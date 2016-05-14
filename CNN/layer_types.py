'''
layer_types.py: This file contains all the layer types that are going to be use in a Convolutional Neural Network.
28.03.2016: Created
'''

#### Libraries
# Standard library
import cPickle
import gzip
import math

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import categorical_crossentropy
from theano.tensor.nnet import softmax, sigmoid
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from learning_functions import sgd, apply_nesterov_momentum, nesterov_momentum, get_or_compute_grads


# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### Define layer types


##########################################
class ConvPoolLayer():
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self,filter_shape,image_shape, poolsize=(2, 2), 
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the 
        filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
	n_in = np.prod(filter_shape[1:])
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/(n_in)), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
                np.zeros((filter_shape[0],),
                dtype=theano.config.floatX), name= 'b',
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
	act_out = self.activation_fn(conv_out + self.b.dimshuffle('x',0,'x','x'))
	pooled_out = downsample.max_pool_2d(input=act_out,ds=self.poolsize,ignore_border=True)
        self.output = pooled_out
        self.output_dropout = self.output # no dropout in the convolutional layers

#######################################
class ConvLayer():
    def __init__(self,filter_shape,image_shape, activation_fn=sigmoid):
        
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.activation_fn=activation_fn
        # initialize weights and biases
	n_in = np.prod(filter_shape[1:])
        n_out = filter_shape[0]*np.prod(filter_shape[2:])
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/(n_in)), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
                np.zeros((filter_shape[0],),
                dtype=theano.config.floatX), name= 'b',
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
	act_out = self.activation_fn(conv_out + self.b.dimshuffle('x',0,'x','x'))
        self.output = act_out
        self.output_dropout = self.output # no dropout in the convolutional layers


############################################
class FullyConnectedLayer():

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/(n_in)), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
	self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)


class PoolLayer():

     def __init__(self,image_shape, poolsize=(2, 2)): 
        self.image_shape = image_shape
        self.poolsize = poolsize

     def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
	pooled_out = downsample.max_pool_2d(input=self.inpt,ds=self.poolsize,ignore_border=True)
        self.output = pooled_out


#######################################
class SoftmaxLayer():

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/(n_in)), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
	self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    #def cost(self, net):
    #    "Return the log-likelihood cost."
    #    return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])
    def cost(self, net):
        "Return the cross entropy cost function"
        return T.mean(categorical_crossentropy(self.output_dropout,net.y))
    def cost_validation(self,net):
    	return T.mean(categorical_crossentropy(self.output,net.y))
    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def false_neg(self, y):
	return T.nonzero(T.gt(y,self.y_out))
    def false_pos(self, y):
	return T.nonzero(T.lt(y,self.y_out))
    def tru_pos(self, y):
	return T.nonzero(T.and_(y,self.y_out))
    def tru_neg(self, y):
	return T.nonzero(T.invert(T.or_(y,self.y_out)))

##########################
class SigmoidLayer():

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/(n_in)), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
	self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the binary cross entropy function"
	return T.mean(binary_crossentropy(self.output_dropout,net.y))

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def false_neg(self, y):
	return T.nonzero(T.gt(y,self.y_out))
    def false_pos(self, y):
	return T.nonzero(T.lt(y,self.y_out))
    def tru_pos(self, y):
	return T.nonzero(T.and_(y,self.y_out))
    def tru_neg(self, y):
	return T.nonzero(T.invert(T.or_(y,self.y_out)))
#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
