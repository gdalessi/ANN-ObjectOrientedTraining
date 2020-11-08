import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

import sourceCode.ANN as neural
from sourceCode.utilities import *

file_options = {
    "path_to_file"                  : "/Users/giuseppedalessio/Dropbox/workPython/plasma_reduction/data", 
    "input_4_file_name"             : "input_case4.csv",
}

### load the training matrix ###
X = readCSV(file_options["path_to_file"], file_options["input_4_file_name"])

### Preprocess the data ###
sigma, X_preprocessed = scale(X, "range", return_scaled_matrix=True)


### Define the autoencoder model ###
model = neural.Autoencoder(X_preprocessed)

model.neurons = 3                           # Reduced dimensionality size
model.activation = "selu"                   # Non-linear activation function
model.n_epochs = 150                        # Number of epochs to ideally do
model.batch_size = 4                        # Batch size to consider
model.patience = 5                          # Patience for early stopping

encoded, X_recovered = model.fit()          # Output: encoded = low-dimensional projection; X_recovered = reconstructed input matrix via AE (scaled space)


#recover the reconstructed matrix in the original space
X_back = unscale(X_recovered, sigma)

### plotting step ###
now = datetime.datetime.now()
try:
    newDirName = "Autoencoder_figures - " + now.strftime("%Y_%m_%d-%H%M")
    os.mkdir(newDirName)
    os.chdir(newDirName)
except FileExistsError:
    newDirName = "Autoencoder_figures 2nd try- " + now.strftime("%Y_%m_%d-%H%M")
    os.mkdir(newDirName)
    os.chdir(newDirName)


#plot the parity in the centered and scaled space for all the variables
matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
a = plt.axes(aspect='equal')
plt.scatter(X_preprocessed.flatten(), X_recovered.flatten(), c='darkgray', label= "$AE\ reconstruction$")
plt.xlabel('$Original\ data$')
plt.ylabel('$Reconstructed\ data\ via\ AE$')
lims = [np.min(X_preprocessed), np.max(X_preprocessed)]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, 'k', label= "$True\ value$")
plt.legend(loc="best")
plt.savefig('Reconstructed_training_AE.png')
plt.show()


#plot the parity for the single variables: Te, Th, e, Ar

Te = X[:,37]
Te_back = X_back[:,37]

matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
a = plt.axes(aspect='equal')
plt.scatter(Te, Te_back, c='darkgray', label= "$AE\ reconstruction$")
plt.xlabel('$Original\ T_{e}\ [K]$')
plt.ylabel('$Reconstructed\ T_{e}\ via\ AE\ [K]$')
lims = [np.min(Te), np.max(Te)]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, 'k', label= "$True\ value$")
plt.legend(loc="best")
plt.savefig('ReconstructedAE_Te_training.png')
plt.show()