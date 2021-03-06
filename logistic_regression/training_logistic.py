'''
Running the code for a logistic regression with the 256*256 data from Pneumonia
'''

from datetime import datetime
########################################
# Look for the data at:
#neumonia_dataset_interson_elDeform_SinNeumEvid_3.pkl
file_name = "../data/neumonia_dataset_interson_elDeform_SinNeumEvid_4.pkl"
########################################
# HyperParameters
startTime = datetime.now()
possible_learning_rate = [1,1.0/10.0,1.0/100.0,1.0/1000.0,1/10000.0] #5
#possible_learning_rate = [1/10000.0] #5
possible_lambda = [5.0,2.0,3.0,1.0,1.0/10.0,1.0/100.0,1.0/1000.0] #7
#possible_lambda = [1.0/1000.0] #7
possible_mini_batch = [100]
possible_dropout =0.5
########################################
# Import libraries
import cPickle
import network_interson
from network_interson import Network
from layer_types import SigmoidLayer, SoftmaxLayer
training_data, validation_data, test_data = network_interson.load_data_shared(filename= file_name)
from network_interson import ReLU
from theano.tensor import tanh

########################################
#Actual training and research of Hyperparameters

for i in range(len(possible_learning_rate)):
	for j in range(len(possible_lambda)):
		for z in range(len(possible_mini_batch)):
			mini_batch_size = possible_mini_batch[z]
			dropout = possible_dropout
			net = Network([
				SoftmaxLayer(n_in=256*256, n_out=2)], mini_batch_size)
			net.SGD(training_data, 50, mini_batch_size, possible_learning_rate[i],validation_data, test_data,lmbda = possible_lambda[j])
			name = 'net_SinEvid4_logistic_%(learning)g_%(lambda)g_%(mini_batch)g_%(dropout)g.pkl' %{"learning": possible_learning_rate[i],"lambda":possible_lambda[j],"mini_batch":mini_batch_size,"dropout":dropout}
			f = file(name,'wb')
			cPickle.dump(net,f,protocol=cPickle.HIGHEST_PROTOCOL)
			f.close()
print datetime.now() - startTime
