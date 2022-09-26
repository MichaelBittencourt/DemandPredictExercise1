import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
#%matplotlib inline #Just to jupyter notebook

class NNeural:
    def __init__(self,
                 name,
                 qtd_inputs = 5,
                 qtd_outputs = 2,
                 qtd_hiden_layers = 2,
                 qtd_neurons_hiden_layers = 20,
                 activation = "sigmoid",
                 optimizer = "adam",
                 loss = "mean_squared_error",
                 metrics = ['mean_squared_error'],
                 activation_final_layer = None,
                 verbosity = False):
        self.setName(name)
        self.setQtdInputs(qtd_inputs)
        self.setQtdOutputs(qtd_outputs)
        self.setQtdHidenLayers(qtd_hiden_layers)
        self.setQtdNeuronsHidenLayers(qtd_neurons_hiden_layers)
        self.setActivation(activation)
        self.setOptimizer(optimizer)
        self.setLoss(loss)
        self.setMetrics(metrics)
        self.setActivationFinalLayer(activation_final_layer if activation_final_layer != None else activation)
        self.setVerbosity(verbosity)
        self.__model = None

    def setName(self, name):
        assert (isinstance(name, str))
        self.__name = name

    def getName(self):
        return self.__name

    def setQtdInputs(self, qtd_inputs):
        assert (isinstance(qtd_inputs, int))
        self.__qtd_inputs = qtd_inputs

    def getQtdInputs(self):
        return self.__qtd_inputs

    def setQtdOutputs(self, qtd_outputs):
        assert (isinstance(qtd_outputs, int))
        self.__qtd_outputs = qtd_outputs

    def getQtdOutputs(self):
        return self.__qtd_outputs

    def setQtdHidenLayers(self, qtd_hiden_layers):
        assert (isinstance(qtd_hiden_layers, int))
        self.__qtd_hiden_layers = qtd_hiden_layers

    def getQtdHidenLayers(self):
        return self.__qtd_hiden_layers

    def setQtdNeuronsHidenLayers(self, qtd_neurons_hiden_layers):
        assert (isinstance(qtd_neurons_hiden_layers, int))
        self.__qtd_neurons_hiden_layers = qtd_neurons_hiden_layers

    def getQtdNeuronsHidenLayers(self):
        return self.__qtd_neurons_hiden_layers

    def setActivation(self, activation):
        self.__activation = activation

    def getActivation(self):
        return self.__activation

    def setOptimizer(self, optimizer):
        self.__optimizer = optimizer

    def getOptimizer(self):
        return self.__optimizer

    def setLoss(self, loss):
        self.__loss = loss

    def getLoss(self):
        return self.__loss

    def setMetrics(self, metrics):
        self.__metrics = metrics

    def getMetrics(self):
        return self.__metrics

    def setActivationFinalLayer(self, activation_final_layer):
        self.__activation_final_layer = activation_final_layer

    def getActivationFinalLayer(self):
        return self.__activation_final_layer

    def setVerbosity(self, verbosity):
        assert(isinstance(verbosity, bool))
        self.__verbosity = verbosity 

    def getVerbosity(self):
        return self.__verbosity

    def __print(self, text):
        if self.getVerbosity():
            print(text)

    def __createModel(self):
        layers = tf.keras.layers
        self.__print("Add Input layer with {} inputs".format(self.getQtdInputs()))
        self.__print("Activation function: {}".format(self.getActivation()))
        model = tf.keras.models.Sequential()
        model.add(layers.Input(self.getQtdInputs()))
        for i in range(self.getQtdHidenLayers()):
            self.__print("Adding hiden layer {layerId} with {qtdNeurons}".format(layerId = i+1, qtdNeurons=self.getQtdNeuronsHidenLayers()))
            model.add(layers.Dense(self.getQtdNeuronsHidenLayers(), activation=self.getActivation()))
        self.__print("Adding output layer with {qtdOutputs} and using {activation} as activation function".format(qtdOutputs = self.getQtdOutputs(), activation = self.getActivationFinalLayer()))
        model.add(layers.Dense(self.getQtdOutputs(), activation=self.getActivationFinalLayer()))

        self.__print("Compiling module with optimizer:{optimizer}, loss:{loss}, metrics:{metrics}".format(optimizer=self.getOptimizer(), loss=self.getLoss(), metrics=self.getMetrics()))
        model.compile(optimizer = self.getOptimizer(),
                      loss = self.getLoss(),
                      metrics = self.getMetrics())
        self.__model = model
 
    def getModel(self):
        if self.__model == None:
            self.__createModel()
        return self.__model

    def train(self, x_train, y_train, epochs=10000):
        self.getModel().fit(x_train, y_train, epochs=epochs)

    def evaluate(self, x_test, y_test):
        return self.getModel().evaluate(x_test, y_test)

    def predict(self, x_test):
        return self.getModel().predict(x_test)

    @staticmethod
    def plot(nneuron_list, x_test, y_test):
        assert(isinstance(nneuron_list, list)), "You need set a list of NNeuron" 
        plt.figure(figsize=(10,8), dpi=110)
        listNames = []
        cyrow = cycle('bgrcmk')
        print(len(nneuron_list))
        for i in range(len(nneuron_list)):
            assert(isinstance(nneuron_list[i], NNeural)), "Each item need be a NNeuron object"
            listNames.append("Potência estimada {}".format(nneuron_list[i].getName()))
            y_pred = nneuron_list[i].predict(x_test)
            #plt.plot(y_pred[:,0], lw=1, color = plt.cm.get_cmap("hsv", i))
            plt.plot(y_pred[:,0], lw=1, color = next(cyrow))
        #plt.plot(y_test[:,0], lw=1, color = plt.cm.get_cmap("hsv", len(nneuron_list)))
        plt.plot(y_test[:,0], lw=1, color = next(cyrow))
        listNames.append("Potência real")
        plt.grid(True)
        plt.xlim([0,91])
        plt.xlabel('Semana')
        plt.ylabel('Potência')
        plt.legend(listNames)

    @staticmethod
    def evaluate_models(nneuron_list, x_test, y_test):
        assert(isinstance(nneuron_list, list)), "You need set a list of NNeuron" 
        ## the tuple return the loss function and all metrics
        listOfResults = []
        metrics = None
        for model in nneuron_list:
            if metrics == None:
                metrics = model.getMetrics()
            else:
                assert(metrics == model.getMetrics()), "The metrics need be equal"
            listOfResults.append(model.evaluate(x_test, y_test))

        print("Metrics head: [loss, " + ", ".join(metrics) + "]")
        for result in listOfResults:
            print ("Metrics result: {}".format(result))
        



