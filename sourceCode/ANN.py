'''
MODULE: ANN.py
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import os
import os.path
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

from .utilities import *


class Architecture:
    def __init__(self, X, Y, *dictionary):
        self.X = X
        self.Y = Y

        self._activation = 'relu'
        self._batch_size = 64
        self._n_epochs = 1000
        self._getNeurons = [1, 1]
        self._dropout = 0
        self._patience = 10
        self._batchNormalization = False
        self._alpha = 0.0001

        self.save_txt = True

        if dictionary:
            settings = dictionary[0]
            try:
                self._center = settings["center"]
            except:
                raise Exception("centering decision not given to the dictionary!")
                exit()
            try:
                self._centering = settings["centering_method"]
            except:
                raise Exception("centering criterion not given to the dictionary!")
                exit()
            try:
                self._scale = settings["scale"]
            except:
                raise Exception("scaling decision not given to the dictionary!")
                exit()
            try:
                self._scaling = settings["scaling_method"]
            except:
                raise Exception("scaling criterion not given to the dictionary!")
                exit()
            try:
                self._activation = settings["activation_function"]
            except:
                raise Exception("activation function not given to the dictionary!")
                exit()
            try:
                self._batch_size = settings["batch_size"]
            except:
                raise Exception("batch size not given to the dictionary!")
                exit()
            try:
                self._n_epochs = settings["number_of_epochs"]
            except:
                raise Exception("number of epochs not given to the dictionary!")
                exit()
            try:
                self._getNeurons = settings["neurons_per_layer"]
            except:
                raise Exception("number of neurons not given to the dictionary!")
                exit()
            try:
                self._dropout = settings["dropout"]
            except:
                raise Exception("dropout not given to the dictionary!")
                exit()
            try:
                self._patience = settings["patience"]
            except:
                raise Exception("patience for early stopping not given to the dictionary!")
                exit()
            try:
                self._batchNormalization = settings["batchNormalization"]
            except:
                raise Exception("batch normalization not given to the dictionary!")
                exit()
            try:
                self._alpha = settings["alpha_LR"]
            except:
                raise Exception("alpha for leaky relu not given to the dictionary!")
                exit()

            if settings["activation_function"] == 'leaky_relu':
                LR = LeakyReLU(alpha=self._alpha)
                LR.__name__ = 'relu'
                self._activation = LR

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, new_activation):
        if new_activation == 'leaky_relu':
            LR = LeakyReLU(alpha=self._alpha)
            LR.__name__ = 'relu'
            self._activation = LR
        else:
            self._activation = new_activation

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    @accepts(object, int)
    def batch_size(self, new_batchsize):
        self._batch_size = new_batchsize

        if self._batch_size <= 0:
            raise Exception("The batch size must be a positive integer. Exiting..")
            exit()

    @property
    def n_epochs(self):
        return self._n_epochs

    @n_epochs.setter
    @accepts(object, int)
    def n_epochs(self, new_epochs):
        self._n_epochs = new_epochs

        if self._n_epochs <= 0:
            raise Exception("The number of epochs must be a positive integer. Exiting..")
            exit()

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, new_value):
        self._dropout = new_value

        if self._dropout < 0:
            raise Exception("The dropout percentage must be a positive integer. Exiting..")
            exit()
        elif self._dropout >= 1:
            raise Exception("The dropout percentage must be lower than 1. Exiting..")
            exit()

    @property
    def patience(self):
        return self._patience

    @patience.setter
    @accepts(object, int)
    def patience(self, new_value):
        self._patience = new_value

        if self._patience < 0:
            raise Exception("Patience for early stopping must be a positive integer. Exiting..")
            exit()

    @property
    def batchNormalization(self):
        return self._batchNormalization

    @batchNormalization.setter
    @accepts(object, bool)
    def batchNormalization(self, new_bool):
        self._batchNormalization = new_bool

    @property
    def getNeurons(self):
        return self._getNeurons

    @getNeurons.setter
    def getNeurons(self, new_vector):
        self._getNeurons = new_vector

    @property
    def alpha_leaky(self):
        return self._alpha

    @alpha_leaky.setter
    def alpha_leaky(self, new_value):
        self._alpha = new_value


    @staticmethod
    def set_environment():
        '''
        This function creates a new folder where all the produced files
        will be saved.
        '''
        import datetime
        import sys
        import os

        now = datetime.datetime.now()
        newDirName = "Train MLP class - " + now.strftime("%Y_%m_%d-%H%M")
        
        try:
            os.mkdir(newDirName)
            os.chdir(newDirName)
        except FileExistsError:
            pass

    @staticmethod
    def write_recap_text(neuro_number, number_batches, activation_specification):
        '''
        This function writes a txt with all the hyperparameters
        recaped, to not forget the settings if several trainings are
        launched all together.
        '''
        text_file = open("recap_training.txt", "wt")
        neurons_number = text_file.write("The number of neurons in the implemented architecture is equal to: {} \n".format(neuro_number))
        batches_number = text_file.write("The batch size is equal to: {} \n".format(number_batches))
        activation_used = text_file.write("The activation function which was used was: "+ str(activation_specification) + ". \n")
        text_file.close()


    @staticmethod
    def preprocess_training(X, centering_decision, scaling_decision, centering_method, scaling_method):

        if centering_decision and scaling_decision:
            mu, X_ = center(X, centering_method, True)
            sigma, X_tilde = scale(X_, scaling_method, True)
        elif centering_decision and not scaling_decision:
            mu, X_tilde = center(X, centering_method, True)
        elif scaling_decision and not centering_decision:
            sigma, X_tilde = scale(X, scaling_method, True)
        else:
            X_tilde = X
            

        return X_tilde


    @staticmethod
    def go_back_to_vector(classification_matrix):
        
        predicted_index_vector = np.empty((classification_matrix.shape[0],), dtype=int)
        for ii in range(classification_matrix.shape[0]):
            predicted_index_vector[ii] = np.argmax(classification_matrix[ii,:])
        
        return predicted_index_vector



class regressor(Architecture):
    def __init__(self, X, Y, *dictionary):
        self.X = X
        self.Y = Y
        
        self._activation_output = 'linear'
        self._loss_function = 'mean_squared_error'
        self._monitor_early_stop = 'mean_squared_error'
        self._learningRate = 0.0001

        self.testProcess = False

        super().__init__(self.X, self.Y, *dictionary)

        if dictionary:
            settings = dictionary[0]
            try:
                self._center = settings["center"]
            except:
                raise Exception("centering decision not given to the dictionary!")
                exit()
            try:
                self._centering = settings["centering_method"]
            except:
                raise Exception("centering criterion not given to the dictionary!")
                exit()
            try:
                self._scale = settings["scale"]
            except:
                raise Exception("scaling decision not given to the dictionary!")
                exit()
            try:
                self._scaling = settings["scaling_method"]
            except:
                raise Exception("scaling criterion not given to the dictionary!")
                exit()
            try:
                self._activation = settings["activation_function"]
            except:
                raise Exception("activation function not given to the dictionary!")
                exit()
            try:
                self._batch_size = settings["batch_size"]
            except:
                raise Exception("batch size not given to the dictionary!")
                exit()
            try:
                self._n_epochs = settings["number_of_epochs"]
            except:
                raise Exception("number of epochs not given to the dictionary!")
                exit()
            try:
                self._getNeurons = settings["neurons_per_layer"]
            except:
                raise Exception("number of neurons per layer not given to the dictionary!")
                exit()
            try:
                self._dropout = settings["dropout"]
            except:
                raise Exception("dropout not given to the dictionary!")
                exit()
            try:
                self._patience = settings["patience"]
            except:
                raise Exception("patience for early stopping not given to the dictionary!")
                exit()
            try:
                self._alpha = settings["alpha_LR"]
            except:
                raise Exception("centering decision not given to the dictionary!")
                exit()
            try:
                self._activation_output = settings["activation_output"]
            except:
                raise Exception("activation output layer not given to the dictionary!")
                exit()
            try:
                self._loss_function = settings["loss_function"]
            except:
                raise Exception("loss function not given to the dictionary!")
                exit()
            try:
                self._monitor_early_stop = settings["monitor"]
            except:
                raise Exception("monitor for early stopping not given to the dictionary!")
                exit()
            try:
                self._learningRate = settings["learning_rate"]
            except:
                raise Exception("initial learning rate not given to the dictionary!")
                exit()
            try:
                self._batchNormalization = settings["batchNormalization"]
            except:
                raise Exception("batch normalization not given to the dictionary!")
                exit()


            if settings["activation_function"] == 'leaky_relu':
                LR = LeakyReLU(alpha=self._alpha)
                LR.__name__ = 'relu'
                self._activation = LR
        
  
        if len(dictionary) > 1:
            self.Z = dictionary[1]
            self.testProcess = True


    @property
    def activationOutput(self):
        return self._activation_output

    @activationOutput.setter
    def activationOutput(self, new_string):
        self._activation_output = new_string


    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, new_value):
        self._loss_function = new_value


    @property
    def monitor_early_stop(self):
        return self._monitor_early_stop

    @monitor_early_stop.setter
    def monitor_early_stop(self, new_value):
        self._monitor_early_stop = new_value


    @property
    def learningRate(self):
        return self._learningRate

    @learningRate.setter
    def learningRate(self, new_value):
        self._learningRate = new_value


    def __set_hard_parameters(self):

        '''
        --- PRIVATE ---
        This function sets all the parameters for the neural network which should not
        be changed during the tuning.
        '''

        self.__path = os.getcwd()
        self.__optimizer = Adam(lr=self._learningRate)
        self.__metrics = "mse"


    def fit_network(self):
        input_dimension = self.X.shape[1]
        try:
            output_dimension = self.Y.shape[1]
        except:
            self.Y = np.reshape(self.Y, (len(self.Y), 1))
            output_dimension = self.Y.shape[1]

        Architecture.set_environment()
        Architecture.write_recap_text(self._getNeurons, self._batch_size, self._activation)
        self.__set_hard_parameters()

        print("Preprocessing training matrix..")
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_tilde, self.Y, test_size=0.2)

        
        self._layers = len(self._getNeurons)


        counter = 0

        self.model = Sequential()
        self.model.add(Dense(self._getNeurons[counter], input_dim=input_dimension, kernel_initializer='normal', activation=self._activation))
        counter += 1
        if self._dropout != 0:
            from tensorflow.python.keras.layers import Dropout
            self.model.add(Dropout(self._dropout))
            print("Dropping out some neurons...")
        elif self._batchNormalization:
            print("Normalization added!")
            self.model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
        else:
            while counter < self._layers:
                print("I'm in the while: {}".format(counter))
                self.model.add(Dense(self._getNeurons[counter], activation=self._activation))
                if self._dropout != 0:
                    self.model.add(Dropout(self._dropout))
                    counter += 1
                elif self._batchNormalization:
                    self.model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
                    counter += 1
                else:
                    counter += 1
        self.model.add(Dense(output_dimension, activation=self._activation_output))
        self.model.summary()

        earlyStopping = EarlyStopping(monitor=self._monitor_early_stop, patience=self._patience, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(filepath=self.__path+ '/best_weights2c.h5', verbose=1, save_best_only=True, monitor=self._monitor_early_stop, mode='min')
        self.model.compile(loss=self._loss_function, optimizer=self.__optimizer, metrics=[self.__metrics])
        history = self.model.fit(self.X_train, self.y_train, batch_size=self._batch_size, epochs=self._n_epochs, verbose=1, validation_data=(self.X_test, self.y_test), callbacks=[earlyStopping, mcp_save])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig('loss_history.eps')
        #plt.show()
        #plt.clf()
        plt.close()

        self.model.load_weights(self.__path + '/best_weights2c.h5')

        counter_saver = 0

        predict_ALL = self.model.predict(self.X_tilde)
        predict_test = self.model.predict(self.X_test)

        if self.save_txt:

            while counter_saver <= self._layers:
                layer_weights = self.model.layers[counter_saver].get_weights()[0]
                layer_biases = self.model.layers[counter_saver].get_weights()[1]
                name_weights = "Weights_HL{}.txt".format(counter_saver)
                name_biases = "Biases_HL{}.txt".format(counter_saver)
                np.savetxt(name_weights, layer_weights)
                np.savetxt(name_biases, layer_biases)

                counter_saver += 1

        return predict_ALL, predict_test, self.y_test



class classifier(Architecture):
    def __init__(self, X, Y, *dictionary):
        self.X = X
        self.Y = Y
        super().__init__(self.X, self.Y, *dictionary)

        if dictionary:
            settings = dictionary[0]

            try:
                self._center = settings["center"]
            except:
                raise Exception("centering decision not given to the dictionary!")
                exit()
            try:
                self._centering = settings["centering_method"]
            except:
                raise Exception("centering criterion not given to the dictionary!")
                exit()
            try:
                self._scale = settings["scale"]
            except:
                raise Exception("scaling decision not given to the dictionary!")
                exit()
            try:
                self._scaling = settings["scaling_method"]
            except:
                raise Exception("scaling criterion not given to the dictionary!")
                exit()
            try:
                self._activation = settings["activation_function"]
            except:
                raise Exception("activation function not given to the dictionary!")
                exit()
            try:
                self._batch_size = settings["batch_size"]
            except:
                raise Exception("batch size not given to the dictionary!")
                exit()
            try:
                self._n_epochs = settings["number_of_epochs"]
            except:
                raise Exception("number of epochs not given to the dictionary!")
                exit()
            try:
                self._getNeurons = settings["neurons_per_layer"]
            except:
                raise Exception("number of neurons not given to the dictionary!")
                exit()
            try:
                self._dropout = settings["dropout"]
            except:
                raise Exception("dropout not given to the dictionary!")
                exit()
            try:
                self._patience = settings["patience"]
            except:
                raise Exception("patience for early stopping not given to the dictionary!")
                exit()
            try:
                self._alpha = settings["alpha_LR"]
            except:
                raise Exception("alpha for leaky relu not given to the dictionary!")
                exit()
            

            if settings["activation_function"] == 'leaky_relu':
                LR = LeakyReLU(alpha=self._alpha)
                LR.__name__= 'relu'
                self._activation= LR

            try:
                self.Xtest = dictionary[1]
            except:
                print("Test not given!!")


    def __set_hard_parameters(self):
        '''
        --- PRIVATE ---
        This function sets all the parameters for the neural network which should not
        be changed during the tuning.
        '''
        self.__activation_output = 'softmax'
        self.__path = os.getcwd()
        self.__monitor_early_stop= 'loss'
        self.__optimizer= 'adam'
        self.__loss_classification= 'categorical_crossentropy'
        self.__metrics_classification= 'accuracy'


    @staticmethod
    def idx_to_labels(idx):
        k = max(idx) +1
        n_observations = len(idx)
        
        labels_matrix = np.zeros((n_observations,k), dtype=int)

        for ii in range(0,n_observations):
            labels_matrix[ii,idx[ii]] = 1

        return labels_matrix


    def fit_network(self):
        self.Y = self.idx_to_labels(self.Y)

        Architecture.set_environment()
        Architecture.write_recap_text(self._getNeurons, self._batch_size, self._activation)
       
        print("Preprocessing training matrix..")
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        X_train, X_test, y_train, y_test = train_test_split(self.X_tilde, self.Y, test_size=0.3)
        input_dimension = self.X.shape[1]
        number_of_classes = self.Y.shape[1]
        self.__set_hard_parameters()

        self._layers = len(self._getNeurons)

        counter = 0

        self.classifier = Sequential()
        self.classifier.add(Dense(self._getNeurons[counter], activation=self._activation, kernel_initializer='random_normal', input_dim=input_dimension))
        counter += 1
        if self._dropout != 0:
            self.classifier.add(Dropout(self._dropout))
            print("Dropping out some neurons...")
        while counter < self._layers:
            self.classifier.add(Dense(self._getNeurons[counter], activation=self._activation))
            if self._dropout != 0:
                self.classifier.add(Dropout(self._dropout))
            counter +=1
        self.classifier.add(Dense(number_of_classes, activation=self.__activation_output, kernel_initializer='random_normal'))
        self.classifier.summary()

        earlyStopping = EarlyStopping(monitor=self.__monitor_early_stop, patience=self._patience, verbose=1, mode='min')
        mcp_save = ModelCheckpoint(filepath=self.__path + '/best_weights.h5', verbose=1, save_best_only=True, monitor='val_loss', mode='min')

        self.classifier.compile(optimizer =self.__optimizer,loss=self.__loss_classification, metrics =[self.__metrics_classification])
        history = self.classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=self._batch_size, epochs=self._n_epochs, callbacks=[earlyStopping, mcp_save])

        # Summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch number')
        plt.legend(['Train', 'Test'], loc='lower right')
        plt.savefig('accuracy_history_class.eps')
        plt.show()

        # Summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig('loss_history_class.eps')
        plt.show()

        counter_saver = 0

        if self.save_txt:

            while counter_saver <= self._layers:
                layer_weights = self.classifier.layers[counter_saver].get_weights()[0]
                layer_biases = self.classifier.layers[counter_saver].get_weights()[1]
                name_weights = "Weights_HL{}.txt".format(counter_saver)
                name_biases = "Biases_HL{}.txt".format(counter_saver)
                np.savetxt(name_weights, layer_weights)
                np.savetxt(name_biases, layer_biases)

                counter_saver +=1

        test = self.classifier.predict(self.X_tilde)
        test_vect = self.go_back_to_vector(test)

        

        return test_vect, self.classifier

    
    
    
    def predict(self):
        prediction_testMat = self.classifier.predict(self.Xtest)
        prediction_testVec = self.go_back_to_vector(prediction_testMat)

        return prediction_testVec



class Autoencoder:
    '''
    X in input must be already SCALED
    '''
    def __init__(self, X):
        self.X = X

        self._n_neurons = 1
        self._activation = 'relu'
        self._batch_size = 64
        self._n_epochs = 1000
        self.save_txt = True

    @property
    def neurons(self):
        return self._n_neurons

    @neurons.setter
    @accepts(object, int)
    def neurons(self, new_number):
        self._n_neurons = new_number

        if self._n_neurons <= 0:
            raise Exception("The number of neurons in the hidden layer must be a positive integer. Exiting..")
            exit()
        elif self._n_neurons >= self.X.shape[1]:
            raise Exception("The reduced dimensionality cannot be larger than the original number of variables. Exiting..")
            exit()

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, new_activation):
        if new_activation == 'leaky_relu':
            LR = LeakyReLU(alpha=0.001)
            LR.__name__= 'relu'
            self._activation= LR
        else:
            self._activation = new_activation

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    @accepts(object, int)
    def batch_size(self, new_batchsize):
        self._batch_size = new_batchsize

        if self._batch_size <= 0:
            raise Exception("The batch size must be a positive integer. Exiting..")
            exit()

    @property
    def n_epochs(self):
        return self._n_epochs

    @n_epochs.setter
    @accepts(object, int)
    def n_epochs(self, new_epochs):
        self._n_epochs = new_epochs

        if self._n_epochs <= 0:
            raise Exception("The number of epochs must be a positive integer. Exiting..")
            exit()

    @property
    def patience(self):
        return self._patience

    @patience.setter
    @accepts(object, int)
    def patience(self, new_value):
        self._patience = new_value


    def __set_hard_parameters(self):
        '''
        --- PRIVATE ---
        This function sets all the parameters for the neural network which should not
        be changed during the tuning.
        '''
        self.__activation_output = 'linear'
        self.__path = os.getcwd()
        self.__monitor_early_stop= 'val_loss'
        self.__optimizer= 'adam'

        self.__loss_function= 'mse'
        self.__metrics= 'mse'



    def set_environment(self):
        '''
        This function creates a new folder where all the produced files
        will be saved.
        '''
        import datetime
        import sys
        import os

        now = datetime.datetime.now()
        newDirName = "trainAE_" + "Neurons=" + str(self._n_neurons) + "_BatchSize=" + str(self._batch_size) + "_Activation=" + self._activation 

        try:
            os.mkdir(newDirName)
            os.chdir(newDirName)
        except FileExistsError:
            newDirName = "trainAE_" + "Neurons=" + str(self._n_neurons) + "_BatchSize=" + str(self._batch_size) + "_Activation=" + self._activation + "_Date=" + now.strftime("%Y_%m_%d-%H%M")
            os.mkdir(newDirName)
            os.chdir(newDirName)


    @staticmethod
    def write_recap_text(neuro_number, number_batches, activation_specification):
        '''
        This function writes a txt with all the hyperparameters
        recaped, to not forget the settings if several trainings are
        launched all together.
        '''
        text_file = open("recap_training.txt", "wt")
        neurons_number = text_file.write("The number of neurons in the implemented architecture is equal to: {} \n".format(neuro_number))
        batches_number = text_file.write("The batch size is equal to: {} \n".format(number_batches))
        activation_used = text_file.write("The activation function which was used was: "+ activation_specification + ". \n")
        text_file.close()


    def fit(self):
        from tensorflow.python.keras.layers import Input, Dense
        from tensorflow.python.keras.models import Model

        self.set_environment()
        Autoencoder.write_recap_text(self._n_neurons, self._batch_size, self._activation)

        input_dimension = self.X.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.X, test_size=0.2)


        self.__set_hard_parameters()

        input_data = Input(shape=(input_dimension,))
        encoded = Dense(self._n_neurons, activation=self._activation)(input_data)
        decoded = Dense(input_dimension, activation=self.__activation_output)(encoded)

        autoencoder = Model(input_data, decoded)

        encoder = Model(input_data, encoded)
        encoded_input = Input(shape=(self._n_neurons,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer=self.__optimizer, loss=self.__loss_function)

        earlyStopping = EarlyStopping(monitor=self.__monitor_early_stop, patience=self._patience, verbose=1, mode='min')
        history = autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=self._n_epochs, batch_size=self._batch_size, shuffle=True, callbacks=[earlyStopping])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig('loss_history.eps')
        #plt.show()
        plt.close()

        encoded_X = encoder.predict(self.X)

        if self.save_txt:
            first_layer_weights = encoder.get_weights()[0]
            first_layer_biases  = encoder.get_weights()[1]

            np.savetxt('AEweightsHL1.txt', first_layer_weights)
            np.savetxt('AEbiasHL1.txt', first_layer_biases)

            
        reconstruction = decoder.predict(encoded_X)
        np.save('Encoded_matrix', encoded_X)
        np.save('Reconstructed_matrix', reconstruction)
        
        autoencoder.save("model_autoencoder.h5")
        encoder.save("model_encoder.h5")
        decoder.save("model_decoder.h5")

        '''
        from tensorflow.python.keras.models import load_model
        model_autoencoder = load_model('autoencoder.h5')
        model_encoder = load_model('encoder.h5')
        model_decoder = load_model('decoder.h5')
        '''

        return encoded_X, reconstruction
