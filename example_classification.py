import numpy as np

import sourceCode.ANN as neural
from sourceCode.utilities import *

file_options = {
    "path_to_file"                  : "/Users/giuseppedalessio/Dropbox/workPython/plasma_reduction/data", 
    "path_to_idx"                   : "/Users/giuseppedalessio/Dropbox/workPython/plasma_reduction/compositionClust_withE",
    

    #Input shape: matrix X (n x p), accounting for n statistical observations of p variables
    "input"                         : "input_case4.csv",
    #Output shape: column vector y (n x 1), where each row contains a class label for an observation of X
    "output"                        : "idx.txt",
    #Input shape: matrix X (m x p), accounting for m statistical observations of p variables
    "test"                          : "input_case5.csv"
}


classifier_options = {
    "center"                    : False,
    "centering_method"          : "mean",
    "scale"                     : False,
    "scaling_method"            : "auto",


    "neurons_per_layer"         : [3],
    "batch_size"                : 1024,
    "number_of_epochs"          : 100,


    "activation_function"       : "relu",
    "alpha_LR"                  : 0.001,


    "batchNormalization"        : False,
    "dropout"                   : 0,
    "patience"                  : 5,        

}

#load the training data: Xtrain, Xtest, labels vector
X = readCSV(file_options["path_to_file"], file_options["input"])
idx = np.genfromtxt(file_options["path_to_idx"] + "/" + file_options["output"])
Xtest = readCSV(file_options["path_to_file"], file_options["test"])

#idx and PVs must be integers
idx = np.int_(idx)

#compute the preprocessing factors for the data [OPTIONAL, only if the data are multivar]
mu, _____ = center(X, "mean", True)
sigma, ______ = scale(X, "auto", True)

#center and scale manually the matrices [OPTIONAL, only if the data are multivar]
Xtrain_preprocessed = center_scale(X, mu, sigma)
Xtest_preprocessed = center_scale(Xtest, mu, sigma)

#train the ANN classifier
#the two input: classifier_options and X_test preprocessed are optional
classification = neural.classifier(Xtrain_preprocessed, idx, classifier_options, Xtest_preprocessed)
class_prediction_training_all, classifierANN = classification.fit_network()

#test on the unseen data matrix, Xtest_preprocessed
pred_testVec = classification.predict()