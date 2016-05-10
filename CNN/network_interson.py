"""
network_interson.py
~~~~~~~~~~~~~~
This a variation from the network3.py file of Michael Nielsen's book "Neural Networks and Deep Learning"

URL: http://neuralnetworksanddeeplearning.com/

We have added a class call ConvLayer, which performs a Convolution with weights W and bias b.
Also this code contains several learning algorithms, which where extracted from Lasagne github page.

URL: https://github.com/Lasagne/Lasagne
In addition, the code within the SGD class includes Early Stopping, which can be change with the variable tolerance (set by default to 8). And we are saving the training and validation loss for further analysis.

21.03.2016 - New a component to calculate the Specificity and Sensitivity.

22.03.2016 - Contingency table added

23.03.2016 Adding a weight initiation with a Gaussian distribution with std: sqrt(2.0/(n_in* n_out) based on: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" by He et al, 2015 URL: http://arxiv.org/pdf/1502.01852v1.pdf

28.03.2016 Separate the learning functions as a separate library: learning_functions.py. Also types of layers was translated into layer_types.py for better debugging

"""


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
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from learning_functions import sgd, apply_nesterov_momentum, nesterov_momentum, get_or_compute_grads

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network_interson.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network_interson.py to set\nthe GPU flag to True."

def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

#### Load the Neumonia data
def load_data_shared(filename="../data/neumonia_dataset_interson_elDeform_0_2.pkl"):
    f = file(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]


##################################################################################

#### Main class used to construct and train networks
class Network():
    
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")  
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            validation_data, test_data, lmbda=0.0,tolerance = 5):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
	
	self.epochs = epochs
	self.eta = eta
	self.lmbda = lmbda
	#self.tolerance = tolerance
	
        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
	#l1_norm = sum([(abs(layer.w)).sum() for layer in self.layers])
        cost2= self.layers[-1].cost(self)+0.5*lmbda*l2_norm_squared/num_training_batches 
	#New version with L1 regularization
	#cost1 = self.layers[-1].cost(self)+lmbda*l1_norm/num_training_batches
        #grads = T.grad(cost, self.params)
        updates = nesterov_momentum(cost2, self.params,eta)	###cost2 <-> cost1
	strikes = 0

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost2, updates=updates,	###cost2 <-> cost1
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y: 
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validation_mb = theano.function(cost2, self.x:validation_x,self.y:validation_y)
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: 
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y: 
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: 
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y: 
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
	validation_predictions = theano.function([i], self.layers[-1].y_out,givens={self.x: 
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]})
	true_output = validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x: 
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
	
	#metrics for net performance
	self.valores_test = []
	self.valores_val = []
	self.cost_train = []
	self.valores_train = []

        # Do the actual training
        best_sensitivity = 0.0
	best_iteration = None
	strikes = 0 
	best_F1 = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0: 
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index) #training
		self.cost_train.append(cost_ij)
	 
        print("Finished training network.")
        print("Best Sensitivity of {0:.4%} obtained at iteration {1}".format(
            best_sensitivity,iteration))

	#self.best_test = test_accuracy
        #print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
