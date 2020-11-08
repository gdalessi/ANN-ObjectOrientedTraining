import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sourceCode.ANN as neural
from sourceCode.utilities import *

file_options = {
    "path_to_file"                  : "/Users/giuseppedalessio/Dropbox/workPython/plasma_reduction/data", 
    

    #Input shape: matrix X (n x p), accounting for n statistical observations of p variables
    "input"                         : "input_case4.csv",

}


regressor_options = {
    "center"                    : False,
    "centering_method"          : "mean",
    "scale"                     : False,
    "scaling_method"            : "auto",


    "neurons_per_layer"         : [15],
    "batch_size"                : 1024,
    "number_of_epochs"          : 100,


    "activation_function"       : "relu",
    "activation_output"         : "linear",
    "alpha_LR"                  : 0.001,

    "loss_function"             : "mse",
    "monitor"                   : "mean_squared_error",
    "learning_rate"             : 1E-4,


    "batchNormalization"        : False,
    "dropout"                   : 0,
    "patience"                  : 5,        

}

#load the training data: Xtrain, Xtest, labels vector
X = readCSV(file_options["path_to_file"], file_options["input"])

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
#the input: regressor_options is optional
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