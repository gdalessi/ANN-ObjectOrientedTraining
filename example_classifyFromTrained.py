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
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *


file_options = {
    "path_to_file"                  : "/Users/giuseppedalessio/Dropbox/workPython/plasma_reduction/data", 
    "path_to_weights"               : "/Users/giuseppedalessio/Dropbox/workPython/plasma_reduction/training_25PVs/class25PVs", 
    "path_to_matrices"              : "/Users/giuseppedalessio/Dropbox/workPython/plasma_reduction/training_25PVs/MGLPCA_25PVs",
    
    "input_train"                   : "input_case4.csv",
    "input_test"                    : "input_case4.csv",

    "grid_test"                     : "x_case4.csv",

    "idx_train"                     : "idx.txt",
    "idx_test"                      : "idx_predicted_ANN.txt",

    "labels_names"                  : "labels_argon.csv",
}

### load data ###
#training matrix
X = readCSV(file_options["path_to_file"], file_options["input_train"])
#new, unobserved, matrix 
Y = readCSV(file_options["path_to_file"], file_options["input_test"])

### these two operations are good for my application only, of course ###
X = X[:,:34]
Y = Y[:,:34]

#distance from the shock
x_coord = readCSV(file_options["path_to_file"], file_options["grid_test"])

### center and scale the data ###
mu = center(X, "mean")
sigma = scale(X, "auto")
X_preprocessed = center_scale(X, mu, sigma)
Y_preprocessed = center_scale(Y, mu, sigma)


### load the trained net ###
#load weights
W1 = np.genfromtxt(file_options["path_to_weights"] + "/Weights_HL0.txt")
W2 = np.genfromtxt(file_options["path_to_weights"] + "/Weights_HL1.txt")
#load biases
b1 = np.genfromtxt(file_options["path_to_weights"] + "/Biases_HL0.txt")
b2 = np.genfromtxt(file_options["path_to_weights"] + "/Biases_HL1.txt")


# load the variables (good only for my case, I did a feature selection step before, that's why)
selected_vars = np.int_(np.genfromtxt(file_options["path_to_matrices"] + "/Selected_variables.txt"))


### pass through the net ###

first_layer = Y_preprocessed[:,selected_vars] @ W1 +b1
mask = np.where(first_layer < 0)
first_layer[mask] = 0


class_layer = first_layer @ W2 + b2

#softmax step
max_class_layer = np.max(class_layer, axis=1)

y = np.empty(class_layer.shape, dtype=float)
y__ = np.empty(class_layer.shape, dtype=float)
for jj in range(0, class_layer.shape[1]):
    y[:,jj] = class_layer[:,jj] - max_class_layer
y_ = np.exp(y)

for jj in range(0, class_layer.shape[0]):
    yo = np.sum(y_, axis=1)


for jj in range(0, y_.shape[1]):
    y__[:,jj] = y_[:,jj]/yo
#end of softmax step

### compute the classification vector: going from (n x k) ---> (n x 1) ###
idx_classification = np.empty((Y.shape[0],), dtype=int)
for ii in range(0, Y.shape[0]):
    idx_classification[ii] = np.argmax(y__[ii,:])


### plot the results ###
matplotlib.rcParams.update({'font.size' : 15, 'text.usetex' : True})
bounds = [1,2,3,4]
fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
sc= axes.scatter(x_coord, Y[:,1], 4, c=idx_classification, marker='s', cmap='gnuplot')
axes.set_xlabel('$Distance\ from\ the\ shock\ [m]$')
axes.set_ylabel('$Ar(0)\ [-]$')
axes.set_xlim(min(x_coord), max(x_coord))
axes.set_ylim(min(Y[:,1]), max(Y[:,1]))
#axes.set_title("$T$ spatial profile along the x-axis")
cb = plt.colorbar(sc, spacing='uniform', ticks=bounds)
cb.set_ticks(ticks=range(5))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()