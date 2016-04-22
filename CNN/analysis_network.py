'''
Analyse network
trying stuff out
'''


# Standard library
import cPickle
import gzip
import os
import scipy.misc
import random
import glob
import operator

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

from potato2 import Network

# Activation functions for neurons
def linear(z): return z

def ReLU(z): return T.maximum(0.0, z)

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

best_networks = []
best_sensitivities = []
best_iterations = []
#best_tests = []
best_parameters = []
#Este codigo abre los networks y los pone en una net
os.chdir("/home/santiago/Desktop/neural-networks-and-deep-learning-master/src/")
for file in glob.glob("net_elDeform2*"):
	f = open(file,'rb')
	print(file)
	try: 
		net = cPickle.load(f)
		f.close()
		#extracting the best sensitivity that network has achieved
		sensitivity = net.sensitivity
		best_iteration, best_sensitivity  = max(enumerate(sensitivity), key=operator.itemgetter(1))
		parameters = [net.epochs, net.eta, net.lmbda]
		if len(best_sensitivities) < 5:
			best_sensitivities.append(best_sensitivity)

			best_networks.append(file)
			best_iterations.append(best_iteration)
			#best_tests.append(best_test)
			best_parameters.append(parameters)
			
		else:
			#Veo si el nuevo network es mejor que alguno de mi lista
			minimum = min(best_sensitivities)
			if best_sensitivity >= minimum:
				#Si es mejor lo cambio por ese elemento
				change_position = best_sensitivities.index(minimum)
				best_sensitivities[change_position] = best_sensitivity
				best_networks[change_position] = file
				best_iterations[change_position] = best_iteration
				#best_tests[change_position] = best_test
				best_parameters[change_position] = parameters

	except EOFError:
		f.close()
	     	print "The file does not exist, exiting..."
	except AttributeError:
		f.close()

h = open('analyze_networks_net_elDeform2.pkl','wb')
cPickle.dump([best_networks,best_sensitivities,best_iterations,best_parameters],h,protocol=cPickle.HIGHEST_PROTOCOL)
h.close()


