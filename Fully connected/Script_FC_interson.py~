'''
Script_FC_interson.py: four layer fully connected neural network, used as a control to compare with the performance of the 17 layer convolutional neural network.
SLG
'''

from datetime import datetime
startTime = datetime.now()
possible_learning_rate = [5.0,3.0,2.0,1.0,1.0/10.0,1.0/100.0,1.0/1000.0,1/10000.0] #8
possible_lambda = [5.0,3.0,2.0,1.0,1.0/10.0,1.0/100.0,1.0/1000.0,1/10000.0] #8
possible_mini_batch = [100]
possible_dropout =0.5
import cPickle
import network_interson
from network_interson import Network
from network_interson import ConvPoolLayer, ConvLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network_interson.load_data_shared()
from network_interson import ReLU
from theano.tensor import tanh


for i in range(len(possible_learning_rate)):
	for j in range(len(possible_lambda)):
		for z in range(len(possible_mini_batch)):
			mini_batch_size = possible_mini_batch[z]
			dropout = possible_dropout
			net = Network([
	FullyConnectedLayer(n_in=256*256, n_out= 1000,activation_fn = ReLU, p_dropout = dropout),
	FullyConnectedLayer(n_in=1000, n_out= 100,activation_fn = ReLU, p_dropout = dropout),
	FullyConnectedLayer(n_in=100, n_out= 20,activation_fn = ReLU, p_dropout = dropout),
	SoftmaxLayer(n_in=20, n_out=2)], mini_batch_size)

			net.SGD(training_data, 100, mini_batch_size, possible_learning_rate[i],validation_data, test_data,lmbda = possible_lambda[j])

			name = 'net_l1_deform_%(learning)g_%(lambda)g_%(mini_batch)g_%(dropout)g.pkl' %{"learning": possible_learning_rate[i],"lambda":possible_lambda[j],"mini_batch":mini_batch_size,"dropout":dropout}
			f = file(name,'wb')
			cPickle.dump(net,f,protocol=cPickle.HIGHEST_PROTOCOL)
			f.close()
print datetime.now() - startTime
