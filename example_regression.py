'''
@Author:
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano
@Contacts:
    giuseppe.dalessio@ulb.ac.be
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sourceCode.ANN as neural
from sourceCode.utilities import *

file_options = {
    "path_to_file"                  : "../OpenMORe/data/reactive_flow/", 

    #Input shape: matrix X (n x p), accounting for n statistical observations of p variables
    "input"                         : "turbo2D.csv",

}


regressor_options = {
    "center"                    : False,        #Center the input matrix before training (boolean) - it is done later in the script, so it is currently set to False.
    "centering_method"          : "mean",       #Set the centering method: available options = "mean" or "min". It must be a string.
    "scale"                     : False,        #Scale the input matrix before training (boolean) - it is done later in the script, so it is currently set to False.
    "scaling_method"            : "auto",       #Set the scaling method: available options = "auto", "range", "vast", or "pareto". It must be a string.


    "neurons_per_layer"         : [15],         #Set the number of layers and the number of neurons per layer. With an input like: [x,y,z] the net will have 3 layers, with "x", "y" and "z" neurons, respectively.
    "batch_size"                : 1024,         #Set the batch size for the ANN training. The input must be an integer.
    "number_of_epochs"          : 100,          #Set the number of epochs before the training stops. The input must be an integer.


    "activation_function"       : "relu",       #Set the activation function for the hidden layers. The two "best" activations are "relu" or "leaky_relu", check the Keras documentation for the others.
    "activation_output"         : "linear",     #Set the activation function for the output layer. It is usually set as "linear". Nevertheless, if the output is \in [0,1], "softmax" can be chosen as well.
    "alpha_LR"                  : 0.001,        #Set the slope for the leaky relu activation, in case the latter is selected as activation function for the hidden layer. In case other activations are chosen, this input is ignored

    "loss_function"             : "mse",        #Set the loss function for the ANN training. "mse" stands for mean squared error, also "mae" (mean absolute error) is used sometimes. Check the Keras documentation.
    "monitor"                   : "mean_squared_error", #Set what has to be monitored for the ANN training convergence. 
    "learning_rate"             : 1E-4,         #Set an initial learning rate. It is only the initial value, because an adaptive optimization is carried out during the training. 


    "batchNormalization"        : False,        #Set if batch normalization has to be used. The input must be a boolean.
    "dropout"                   : 0,            #Set which percentage of HL neurons has to be dropped out. This is a method to avoid overfitting. If set to '0', no neurons will be dropped out during the training
    "patience"                  : 5,            #Set the number of epochs which has to be waited before early stopping is activated. If "x" is given as input (it has to be an integer), if the mse of the validation set does not decrease for "x" epochs, the training will be stopped to avoid overfitting.

}

#load the training data: 
X = readCSV(file_options["path_to_file"], file_options["input"])

#in this case, the input and the output to the net are contained in the same matrix
input_ = X[:,:33]
output_ = X[:,34]

#compute the preprocessing factors for the data [OPTIONAL, only if the data are multivar]
muIN, _____ = center(input_, "min", True)
sigmaIN, ______ = scale(input_, "range", True)

muOUT, _____ = center(output_, "min", True)
sigmaOUT, ______ = scale(output_, "range", True)


#center and scale manually the matrices [OPTIONAL, only if the data are multivar]
input_train_preprocessed = center_scale(input_, muIN, sigmaIN)
output_train_preprocessed = center_scale(output_, muOUT, sigmaOUT)


#train the ANN regressor
#the input matrix is split inside the class in input_training (70% of the total training observations) and input_validation (30% of the total training observations)
regression = neural.regressor(input_train_preprocessed, output_train_preprocessed, regressor_options)
predictedInput, predictedTestBatch, trueTestBatch = regression.fit_network()


#plot the parity for all the predicted test batch
matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
a = plt.axes(aspect='equal')
plt.scatter(trueTestBatch.flatten(), predictedTestBatch.flatten(), c='darkgray', label= "$ANN\ prediction$")
plt.xlabel('$Original\ data$')
plt.ylabel('$Predicted\ data\ via\ ANN$')
lims = [np.min(trueTestBatch), np.max(trueTestBatch)]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, 'k', label= "$True\ value$")
plt.legend(loc="best")
plt.savefig('predictedTest_ANN.png')
plt.show()