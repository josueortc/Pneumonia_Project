'''
cross_validation_sklearn:
This is an implementation of a custom Convolutional Neural Network using the Scikit-Learn API for optimization
'''
from __future__ import print_function
import numpy as np


import cPickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1l2, l1
from sklearn import preprocessing

def load_data_p(number):
	'''
	This code lodes the data from the Pneumonia
	dataset into the model
	'''
	
	f = open('neumonia_dataset_interson_keras_alldata10_{0}.pkl'.format(number),'rb')
	data = cPickle.load(f)
	f.close()
	training_data = data[0]
	validation_data = data[1]
	test_data = data[2]

	label_t = training_data[1]
	data_t = training_data[0]

	label_v = validation_data[1]
	data_v = validation_data[0]
	
	label_tt = test_data[1]
	data_tt = test_data[0]

	return (data_t,label_t),(data_v,label_v),(data_tt,label_tt)



# input image dimensions
img_rows, img_cols = 256, 256


 

#load training data and do basic data normalization
number_db = 9
(X_train, y_train), (X_val, y_val), (X_test,y_test) = load_data_p(number_db)
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
X_val /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'val samples')
print(X_test.shape[0], 'test samples')


def make_model(dropout, nb_filters, nb_conv, nb_pool,weight_initiation,activation_function,l1_reg,l2_reg):
    '''Creates model comprised of 2 convolutional layers followed by dense layers
    dense_layer_sizes: List of layer sizes. This list has one number for each layer
    nb_filters: Number of convolutional filters in each convolutional layer
    nb_conv: Convolutional kernel size
    nb_pool: Size of pooling area for max pooling
    '''
    model = Sequential()

    model.add(Convolution2D(8, 8, 8,
		                border_mode='valid',
		                input_shape=(1, img_rows, img_cols),subsample = (4,4),W_regularizer=l1l2(l1 = l1_reg,l2=l2_reg),b_regularizer=l1l2(l1 = l1_reg,l2=l2_reg),init=weight_initiation))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout))
  
    model.add(Convolution2D(8, 5, 5,W_regularizer=l1l2(l1 = l1_reg,l2=l2_reg),b_regularizer=l1l2(l1 = l1_reg,l2=l2_reg),subsample = (2,2),init=weight_initiation))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout))

    model.add(Convolution2D(16, nb_conv, nb_conv,W_regularizer=l1l2(l1=l1_reg,l2=l2_reg),b_regularizer=l1l2(l1 = l1_reg,l2=l2_reg),init=weight_initiation))
    model.add(Activation(activation_function))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(5,W_regularizer=l1l2(l1 = l1_reg,l2=l2_reg),b_regularizer=l1l2(l1 = l1_reg,l2=l2_reg),init=weight_initiation))
    model.add(Activation(activation_function))
    model.add(Dropout(dropout))

    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    return model

#Hyperparameters for tuning
weight_initiation = ['he_normal','glorot_normal']#2
activation_functions = ['relu']#1
l1_reg = [1.0,0.1,0.01,0.001,0.0]#5
l2_reg = [1.0,0.1,0.01,0.001,0.0]#5
dropout = [0.0,0.25,0.5,0.7]#4

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
my_classifier = KerasClassifier(make_model, batch_size = 32)
validator = GridSearchCV(my_classifier,
                         param_grid={'dropout': dropout,'weight_initiation':weight_initiation,
                         	     'l1_reg':l1_reg,
                         	     'l2_reg':l2_reg,
                                     'activation_function':activation_functions,
                                     'nb_epoch': [40],
                                     'nb_filters': [8],
                                     'nb_conv': [3],
                                     'nb_pool': [2]},
                         scoring='precision',
			 cv = 5,
                         n_jobs=1)
                         



validator.fit(X_train, y_train,show_acuraccy = True,callbacks = early_stopping)

print('The parameters of the best model are: ')
print(validator.best_params_)

# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_val, y_val)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
