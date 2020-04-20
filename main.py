'''
PROGRAM: main.py
​
@Author:
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano
​
@Contacts:
    giuseppe.dalessio@ulb.ac.be
​
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be
'''

import ANN as neural
from utilities import *

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "input_file_name"           : "X_zc.csv",
    "output_file_name"          : "Y_zc.csv",
}

training_options = {
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    "neurons_per_layer"         : [256, 512],
    "batch_size"                : 64,
    "number_of_epochs"          : 1000,

    "activation_function"       : "leaky_relu",
    "alpha_LR"                  : 0.0001,
    "activation_output"         : "softmax",

    "batchNormalization"        : True,
    "dropout"                   : 0,
    "patience"                  : 10,        
}


X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
Y = readCSV(file_options["path_to_file"], file_options["output_file_name"])

model = neural.regressor(X, Y, training_options)
predicted_Y = model.fit_network()
predictedTest, trueTest = model.predict()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(Y.flatten(), predicted_Y.flatten(), s=2, edgecolors='black', linewidths=0.1)
plt.xlabel('Y_zc')
plt.ylabel('Y_pred')
lims = [np.min(predicted_Y), np.max(predicted_Y)]
lims2 = [np.min(Y), np.max(Y)]
# plt.xlim(lims)
# plt.ylim(lims)
_ = plt.plot(lims2, lims2, 'r')
plt.show()