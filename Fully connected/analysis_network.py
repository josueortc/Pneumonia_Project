'''
analysis_network.py for Fully connected net
Goes through the .pkl files;
Issue: AttributeError: Network instance has no attribute 'epochs' -> in parameters, replace net.epochs with 100
'''

# Standard library
import cPickle
import gzip
import os
import scipy.misc
import random
import glob

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

import network_interson
from network_interson import Network

# Activation functions for neurons
def linear(z): return z

def ReLU(z): return T.maximum(0.0, z)

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

best_networks = []
best_validations = []
best_iterations = []
best_tests = []
best_parameters = []
#Este codigo abre los networks y los pone en una net
os.chdir("/home/santiago/Desktop/neural-networks-and-deep-learning-master/Fully connected")
for file in glob.glob("*.pkl"):
	f = open(file,'rb')
	print(file)
	try: 
		net = cPickle.load(f)
		f.close()
		#Ahora el siguiente codigo va a extraer el best validation
		best_validation = net.best_validation
		best_iteration = net.best_iteration
		best_test = net.best_test
		parameters = [net.epochs, net.eta, net.lmbda, net.tolerance]
		if len(best_validations) < 5:
			best_validations.append(best_test)
			best_networks.append(file)
			best_iterations.append(best_iteration)
			best_tests.append(best_test)
			best_parameters.append(parameters)
			
		else:
			#Veo si el nuevo network es mejor que alguno de mi lista
			minimum = min(best_validations)
			if best_test > minimum:
				#Si es mejor lo cambio por ese elemento
				change_position = best_validations.index(minimum)
				best_validations[change_position] = best_validation
				best_networks[change_position] = file
				best_iterations[change_position] = best_iteration
				best_tests[change_position] = best_test
				best_parameters[change_position] = parameters
	except EOFError:
		f.close()
	     	print "The file does not exist, exiting..."
h = open('analyze_networks_deform_l2.pkl','wb')
cPickle.dump([best_networks,best_validations,best_iterations,best_tests,best_parameters],h,protocol=cPickle.HIGHEST_PROTOCOL)
h.close()


