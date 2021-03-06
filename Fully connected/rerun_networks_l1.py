#Este codigo va a correr cada uno de los mejores 8 networks 10 veces y sacar el promedio de sus performances
#Libraries
import cPickle
import network_interson
import re
from network_interson import Network
from network_interson import ConvPoolLayer, ConvLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network_interson.load_data_shared()
from network_interson import ReLU
from theano.tensor import tanh
import numpy as np
#Open the file of the analyzed networks
f = open('networks/FC_l1/analyze_networks_deform_l1.pkl','rb') #takes pkls from network folder
networks = cPickle.load(f)
f.close()
names = networks[0]
parameters = networks[4]
averages_validation = []
std_validation = []
averages_best_test = []
std_test = []
names_average = []
for x in xrange(len(names)):
	#Estoy extrayendo los hyperparametros del nombre del archivo
	per_network_best_validation = []
	per_network_best_test = []
	hyperparameters = parameters[x]
	epochs = hyperparameters[0]
	learning_rate = hyperparameters[1]
	lmbda_m = hyperparameters[2]
	mini_batch_size = 100
	dropout = 0.5
	tolerance = hyperparameters[3]
	for i in range(10):
		net = Network([
				ConvLayer(image_shape=(mini_batch_size, 1, 256,256), 
					      filter_shape=(8, 1, 3, 3), 
					      activation_fn=ReLU),
				ConvPoolLayer(image_shape=(mini_batch_size, 8, 254,254), 
					      filter_shape=(8, 8, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 8, 126,126), 
					      filter_shape=(16, 8, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 16, 62,62), 
					      filter_shape=(32, 16, 3, 3), 
					      activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 32, 60,60), 
					      filter_shape=(32, 32, 3, 3), 
					      activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 32, 58,58), 
					      filter_shape=(32, 32, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 32, 28,28), 
					      filter_shape=(64, 32, 3, 3), 
					      activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 64, 26,26), 
					      filter_shape=(64, 64, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 64, 12,12), 
					      filter_shape=(128, 64, 3, 3), 
					      activation_fn=ReLU),
				ConvPoolLayer(image_shape=(mini_batch_size, 128, 10,10), 
					      filter_shape=(128, 128,3, 3), 
					      poolsize=(2, 2),activation_fn = ReLU),
				FullyConnectedLayer(n_in=128*4*4, n_out= 5,activation_fn = ReLU, 					p_dropout = dropout),
				SoftmaxLayer(n_in=5, n_out=2)], mini_batch_size)
		net.SGD(training_data, 100, mini_batch_size, learning_rate,validation_data, test_data,lmbda = lmbda_m, tolerance=tolerance)
		name = 'net_l1_rerun_interson_%(learning)g_%(lambda)g_%(mini_batch)g_%(dropout)g_%(trial)g.pkl' %{"learning": learning_rate,"lambda":lmbda_m,"mini_batch":mini_batch_size,"dropout":dropout,"trial":i}
		per_network_best_validation.append(net.best_validation)
		per_network_best_test.append(net.best_test)
		f = file(name,'wb')
		cPickle.dump(net,f,protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
	#Get the average of each network for validation and testing and see if it converges
	averages_validation.append(np.mean(per_network_best_validation))
	std_validation.append(np.std(per_network_best_validation))
	averages_best_test.append(np.mean(per_network_best_test))
	std_test.append(np.std(per_network_best_test))
	names_average.append(name)
h = file('average_l1_interson_deform_rerun.pkl','wb')
cPickle.dump([names_average,averages_validation,std_validation,averages_best_test,std_test],h,protocol=cPickle.HIGHEST_PROTOCOL)
h.close()
