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
import datetime
import os
import sourceCode.ANN as neural
from sourceCode.utilities import *

########## SETTINGS ##########

file_options = {
    "path_to_file"                  : "../OpenMORe/data/reactive_flow/", 
    "input_4_file_name"             : "turbo2D.csv",
}

### load the training matrix ###
X = readCSV(file_options["path_to_file"], file_options["input_4_file_name"])

### Preprocess the data ###
sigma, X_preprocessed = scale(X, "range", return_scaled_matrix=True)


### Define the autoencoder model ###
model = neural.Autoencoder(X_preprocessed)

model.neurons = 3                           # Reduced dimensionality size
model.activation = "relu"                   # Non-linear activation function
model.n_epochs = 15                        # Number of epochs to ideally do
model.batch_size = 2048                        # Batch size to consider
model.patience = 5                          # Patience for early stopping

encoded, X_recovered = model.fit()          # Output: encoded = low-dimensional projection; X_recovered = reconstructed input matrix via AE (scaled space)

########## END OF SETTINGS ##########

#recover the reconstructed matrix in the original space
X_back = unscale(X_recovered, sigma)


### plotting step ###
now = datetime.datetime.now()
try:
    newDirName = "figures " + now.strftime("%Y_%m_%d-%H%M")
    os.mkdir(newDirName)
    os.chdir(newDirName)
except FileExistsError:
    newDirName = "figures" + now.strftime("%Y_%m_%d-%H%M")
    os.mkdir(newDirName)
    os.chdir(newDirName)


#plot the parity in the centered and scaled space for all the variables
matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
a = plt.axes(aspect='equal')
plt.scatter(X_preprocessed.flatten(), X_recovered.flatten(), c='darkgray',  s=10, edgecolors='b', label= "$AE\ reconstruction$")
plt.xlabel('$Original\ data$')
plt.ylabel('$Reconstructed\ data\ via\ AE$')
lims = [np.min(X_preprocessed), np.max(X_preprocessed)]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, 'k', label= "$True\ value$")
plt.legend(loc="best")
plt.savefig('Reconstructed_training_AE.png', dpi=300)
plt.show()


#plot the parity for the single variables: Te, Th, e, Ar

T = X[:,0]
T_back = X_back[:,0]

matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
a = plt.axes(aspect='equal')
plt.scatter(T, T_back, c='darkgray', s=15, edgecolors='b', label= "$AE\ reconstruction$")
plt.xlabel('$Original\ T\ [K]$')
plt.ylabel('$Reconstructed\ T\ via\ AE\ [K]$')
lims = [np.min(T), np.max(T)]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, 'k', label= "$True\ value$")
plt.legend(loc="best")
plt.savefig('ReconstructedAE_T_training.png', dpi=300)
plt.show()
