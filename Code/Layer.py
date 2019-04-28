'''
    name: Layer.py
    purpose: Create a layer of Perceptrons.
    date: March 11, 2019
'''

from Perceptron import Perceptron
from copy import deepcopy

class Layer:
    def __init__(self, output_layer_flag, layer_length, layer_weight, layer_bias):
        # print("__init__ called!")
        self.error = 0
        self.output_layer_flag = False
        self.layer_length = 0
        self.layer_weight = []
        self.layer_bias = []
        self.perceptrons = []
        self.__check_output_layer_flag(deepcopy(output_layer_flag))
        self.__check_layer_length(deepcopy(layer_length))
        self.__check_layer_weight(deepcopy(layer_weight))
        self.__check_layer_bias(deepcopy(layer_bias))
        self.littleE_vector = [0] * self.layer_length
        self.output_vector = [0] * self.layer_length
        self.__create_perceptrons()
    # get error vector
    def get_error_vector(self, desired_output):
        self.__check_desired_output(deepcopy(desired_output))
        try:
            if self.error == 1:
                raise AttributeError("1 or more variables not set properly.")
            # check if output layer
            if self.output_layer_flag == True:
                # calculate error associated with all values in output layer
                for i in range(self.layer_length):
                    self.littleE_vector[i] = desired_output[i] - self.output_vector[i]
            return self.littleE_vector
        except:
            self.error = 1
            raise
            # return empty list if error
            return []
    # get layer output vector
    def get_layer_output_vector(self, inp):
        try:
            if self.error == 1:
                raise AttributeError("1 or more variables not set properly.")
            # calculate all activation values for perceptrons in layer, store in output vector
            for i in range(self.layer_length):
                self.perceptrons[i].calc_activity(inp)
                self.perceptrons[i].calc_activation()
                self.output_vector[i] = self.perceptrons[i].activation
            return self.output_vector
        except:
            self.error = 1
            raise
            return []
    # get layer length
    def get_layer_length(self):
        return self.layer_length
    # set output layer delta values
    def set_output_layer_delta_values(self):
        try:
            if self.error == 1:
                raise AttributeError("1 or more variables not set properly.")
            # check if output layer
            if self.output_layer_flag == True:
                for i in range(self.layer_length):
                    # error value gets multiplied by o(1-o) within perceptron
                    self.perceptrons[i].set_delta(self.littleE_vector[i])
        except:
            self.error = 1
            raise
    # set hidden layer delta values
    def set_hidden_layer_delta_values(self, l):
        try:
            if self.error == 1:
                raise AttributeError("1 or more variables not set properly.")
            # check if hidden layer
            if self.output_layer_flag == False:
                # make sure that each of the perceptrons in l can take in current layer input
                for j in range(l.get_layer_length()):
                    if l.perceptrons[j].wlines != self.layer_length:
                        raise AttributeError("Output layer not compatible with current layer.")
                # calculate delta for each perceptron using summation delta*weight
                for i in range(self.layer_length):
                    delta_weight_summation = 0
                    for j in range(l.get_layer_length()):
                        delta_weight_summation += (l.perceptrons[j].delta*l.perceptrons[j].weights[i])
                    # set delta for hidden layer perceptron
                    self.perceptrons[i].set_delta(delta_weight_summation)
        except:
            self.error = 1
            raise
    # calculate layer delta weights
    def calc_layer_delta_weights(self, eta):
        try:
            if self.error == 1:
                raise AttributeError("1 or more variables not set properly.")
            # set delta weights
            for i in range(self.layer_length):
                self.perceptrons[i].set_delta_weights(deepcopy(eta))
        except:
            self.error = 1
            raise
    # update layer weights
    def update_layer_weights(self):
        try:
            if self.error == 1:
                raise AttributeError("1 or more variables not set properly.")
            # update weights
            for i in range(self.layer_length):
                self.perceptrons[i].update_weights()
        except:
            self.error = 1
            raise
    def __check_output_layer_flag(self, output_layer_flag):
        # print("check output layer flag!")
        self.error = 1
        try:
            if output_layer_flag == False or output_layer_flag == True:
                self.error = 0
                self.output_layer_flag = output_layer_flag
            else:
                raise ValueError("Output layer flag must be 0 or 1.")
        except:
            raise
    def __check_layer_length(self, layer_length):
        # print("check layer length!")
        self.error = 1
        try:
            if layer_length > 0:
                self.layer_length = layer_length
                self.error = 0
            else:
                raise ValueError("Layer length should be greater than 0.")
        except:
            raise
    def __check_layer_weight(self, layer_weight):
        # print("check layer weight!")
        self.error = 1
        try:
            wlen = len(layer_weight)
            if wlen != self.layer_length:
                raise ValueError("Weight length should be equal to layer length.")
            else:
                self.layer_weight = layer_weight
                self.error = 0
        except:
            raise
    def __check_layer_bias(self, layer_bias):
        # print("check layer bias!")
        self.error = 1
        try:
            blen = len(layer_bias)
            if blen != self.layer_length:
                raise ValueError("Bias length should be equal to layer length.")
            else:
                for i in range(blen):
                    layer_bias[i] = float(layer_bias[i])
                self.layer_bias = layer_bias
                self.error = 0
        except:
            raise
    def __create_perceptrons(self):
        # print("check perceptrons!")
        try:
            if self.error == 1:
                raise AttributeError("1 or more variables not set properly.")
            else:
                for i in range(self.layer_length):
                    p = Perceptron(self.layer_weight[i], self.layer_bias[i])
                    self.perceptrons.append(p)
        except:
            raise
    def __check_desired_output(self, desired_output):
        # print("check desired output!")
        self.error = 1
        try:
            dlen = len(desired_output)
            if dlen != self.layer_length:
                raise ValueError("Desired output length should be equal to layer length.")
            else:
                self.error = 0
        except:
            raise