#Este codigo va a correr cada uno de los mejores 8 networks 10 veces y sacar el promedio de sus performances
#Libraries
import cPickle
import network_interson
import re
from network_interson import Network
from layer_types import ConvPoolLayer, ConvLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network_interson.load_data_shared()
from network_interson import ReLU
from theano.tensor import tanh
import numpy as np
#Open the file of the analyzed networks
f = open('analyze_networks_net_elDeform2.pkl','rb')
networks = cPickle.load(f)
f.close()
names = networks[0]
parameters = networks[3]
averages_sensitivity = []
std_sensitivity = []
#averages_best_test = []
#std_test = []
names_average = []
for x in xrange(len(names)):
	#Estoy extrayendo los hyperparametros del nombre del archivo
	per_network_best_sensitivity = []
	#per_network_best_test = []
	hyperparameters = parameters[x]
	epochs = hyperparameters[0]
	learning_rate = hyperparameters[1]
	lmbda_m = hyperparameters[2]
	mini_batch_size = 100
	dropout = 0.5
	#tolerance = hyperparameters[3]
	for i in range(10):
		net = Network([
				ConvLayer(image_shape=(mini_batch_size, 1, 256,256), 
					      filter_shape=(8, 1, 3, 3), 
					      activation_fn=ReLU),
				ConvPoolLayer(image_shape=(mini_batch_size, 8, 254,254), 
					      filter_shape=(8, 8, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 8, 126,126), 
					      filter_shape=(8, 8, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 8, 62,62), 
					      filter_shape=(16, 8, 3, 3), 
					      activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 16, 60,60), 
					      filter_shape=(16, 16, 3, 3), 
					      activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 16, 58,58), 
					      filter_shape=(16, 16, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 16, 28,28), 
					      filter_shape=(32, 16, 3, 3), 
					      activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 32, 26,26), 
					      filter_shape=(32, 32, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 32, 12,12), 
					      filter_shape=(32, 32, 3, 3), 
					      activation_fn=ReLU),
				ConvPoolLayer(image_shape=(mini_batch_size, 32, 10,10), 
					      filter_shape=(32, 32,3, 3), 
					      poolsize=(2, 2),activation_fn = ReLU),
				FullyConnectedLayer(n_in=32*4*4, n_out= 10,activation_fn = ReLU, p_dropout = dropout),
				FullyConnectedLayer(n_in=10, n_out= 5,activation_fn = ReLU, p_dropout = dropout),
				SoftmaxLayer(n_in=5, n_out=2)], mini_batch_size)
		net.SGD(training_data, 50, mini_batch_size, possible_learning_rate[i],validation_data, test_data,lmbda = possible_lambda[j])
		name = 'net_elDeform2_rerun_%(learning)g_%(lambda)g_%(mini_batch)g_%(dropout)g_%(trial)g.pkl' %{"learning": learning_rate,"lambda":lmbda_m,"mini_batch":mini_batch_size,"dropout":dropout,"trial":i}
		per_network_best_sensitivity.append(net.best_sensitivity)
		#per_network_best_test.append(net.best_test)
		f = file(name,'wb')
		cPickle.dump(net,f,protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
	#Get the average of each network for validation and testing and see if it converges
	averages_sensitivity.append(np.mean(per_network_best_sensitivity))
	std_sensitivity.append(np.std(per_network_best_sensitivity))
	#averages_best_test.append(np.mean(per_network_best_test))
	#std_test.append(np.std(per_network_best_test))
	names_average.append(name)
h = file('average_elDeform2_rerun_.pkl','wb')
cPickle.dump([names_average,averages_sensitivity,std_sensitivity],h,protocol=cPickle.HIGHEST_PROTOCOL)
h.close()



