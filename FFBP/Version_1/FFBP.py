import math
import random
from copy import deepcopy
import numpy as np
from datetime import datetime

class Perceptron:
    def __init__(self, num_inputs, weights, bias, eta, output_layer_flag):
        self.num_inputs = deepcopy(num_inputs)
        # length of weights should match num inputs
        self.weights = deepcopy(weights)
        self.bias = deepcopy(bias)
        self.eta = deepcopy(eta)
        self.output_layer = deepcopy(output_layer_flag)
        self.activity = None
        self.activation = None

    def generate_output(self, inputs):
        self.activity = 0
        for i in range(0, self.num_inputs):
            input = inputs[i]
            weight = self.weights[i]
            self.activity += (self.weights[i] * inputs[i])

        exponent = self.activity + self.bias
        denominator = 1 + np.exp(-1 * exponent)
        self.activation = 1 / denominator

        return self.activation

    def generate_error(self, desired_output):
        error = desired_output - self.activation

        return error


class Layer:
    def __init__(self, num_input_nodes, num_nodes, initial_weights, output_layer_flag, bias, eta):
        self.num_input_nodes = num_input_nodes
        self.num_nodes = num_nodes
        self.weights = initial_weights
        self.output_layer_flag = output_layer_flag
        self.eta = eta
        self.perceptrons = []
        self.delta_values = None

        for i in range(0, self.num_nodes):
            self.perceptrons.append(Perceptron(self.num_input_nodes, self.weights[i], bias, self.eta, self.output_layer_flag))

    def generate_outputs(self, inputs):
        outputs = []
        for i in range(0, self.num_nodes):
            perceptron = self.perceptrons[i]
            output = perceptron.generate_output(inputs)
            outputs.append(output)

        return outputs

    def generate_error(self, desired_output):
        if self.num_nodes == 1 and self.output_layer_flag:
            perceptron = self.perceptrons[0]
            error = perceptron.generate_error(desired_output)
            return [error]
        else:
            print("Unhandled case for generating error")

    def set_output_delta_values(self, desired_output):
        if self.num_nodes == 1 and self.output_layer_flag:
            perceptron = self.perceptrons[0]
            error = perceptron.generate_error(desired_output)

            self.delta_values = [(desired_output - perceptron.activation) * (1 - perceptron.activation) * perceptron.activation]
            return self.delta_values
        else:
            print("Unhandled case for generating error")

    def set_hidden_delta_values(self, next_layer):
        if self.num_nodes > 1 and not self.output_layer_flag:
            self.delta_values = []
            for i in range(0, len(self.perceptrons)):
                perceptron = self.perceptrons[i]
                output_initial_weight = next_layer.perceptrons[0].weights[i]
                output_delta = next_layer.delta_values[0]
                delta_value = (1 - perceptron.activation) * perceptron.activation * (output_delta * output_initial_weight)
                self.delta_values.append(delta_value)

            return self.delta_values
        else:
            print("Unhandled case for generating error")

    def update_weights(self, inputs):
        updated_weights = []
        for i in range(0, len(self.weights)):
            node_weights = []
            input = inputs[i]
            for j in range(0, self.num_input_nodes):
                updated_weight = self.weights[i][j] + (self.eta*self.delta_values[i]*input)
                node_weights.append(updated_weight)
            updated_weights.append(node_weights)
        self.weights = updated_weights

        # update perceptron weights
        for i in range(0, len(self.perceptrons)):
            perceptron = self.perceptrons[i]
            perceptron.weights = updated_weights[i]

        return updated_weights

    def update_bias(self):
        # update perceptron bias
        updated_biases = []
        for i in range(0, len(self.perceptrons)):
            perceptron = self.perceptrons[i]
            original_bias = perceptron.bias
            perceptron.bias = original_bias + (self.eta * self.delta_values[i] * 1)
            updated_biases.append(perceptron.bias)

        return updated_biases
    
    
def runNeuralNetwork(num_input_nodes):
    # inputs = [1, 2]
    # desired_output = [0.7]
    eta = 1.0
    bias = 0

    num_hidden_nodes = math.ceil(num_input_nodes*.66)

    weights_hidden = []
    weights_output = []

    for i in range(0, num_hidden_nodes):
        hidden_node_weights = []
        for j in range(0, num_input_nodes):
            hidden_node_weights.append(random.uniform(0,1))
        weights_hidden.append(hidden_node_weights)

    output_node_weight = []
    for i in range(0, num_hidden_nodes):
        output_node_weight.append(random.uniform(0, 1))
    weights_output = [output_node_weight]

    # epochs = 15
    # instantiate layers
    hidden_layer = Layer(num_input_nodes, num_hidden_nodes, weights_hidden, False, bias, eta)
    output_layer = Layer(num_hidden_nodes, 1, weights_output, True, bias, eta)

    # print("number of input nodes:", num_input_nodes)
    # print("number of hidden nodes:", num_hidden_nodes)

    input_nodes = []
    for i in range(0, num_input_nodes):
        input_nodes.append(random.uniform(0, 1))

    assignment_inputs = [input_nodes]
    assignment_desired_output = [1]

    for j in range(0, len(assignment_inputs)):
        inputs = assignment_inputs[j]
        desired_output = assignment_desired_output[j]

        # forward
        hidden_output = hidden_layer.generate_outputs(inputs)
        # print("hidden layer output:", hidden_output)
        output_output = output_layer.generate_outputs(hidden_output)
        # print("output layer output:", output_output)
        error = output_layer.generate_error(desired_output)
        # print("little e:", error)

        big_e = 0.5 * error[0] ** 2
        # print("big e:", big_e)

        # backward
        # set delta values
        output_delta_values = output_layer.set_output_delta_values(desired_output)
        # print("output layer updated weights:", output_delta_values)
        hidden_delta_values = hidden_layer.set_hidden_delta_values(output_layer)
        # print("output layer updated weights:", hidden_delta_values)

        # update weights
        output_updated_weights = output_layer.update_weights(hidden_output)
        # print("output layer updated weights:", output_updated_weights)
        hidden_updated_weights = hidden_layer.update_weights(inputs)
        # print("hidden layer updated weights:", hidden_updated_weights)

        # update biases
        output_updated_bias = output_layer.update_bias()
        # print("output layer updated bias:", output_updated_bias)
        hidden_updated_bias = hidden_layer.update_bias()
        # print("hidden layer updated bias:", hidden_updated_bias)


if __name__ == "__main__":
    n = 2

    while n < 128:
        print("N:", n)

        time_format = '%H:%M:%S'
        sum_time = 0

        for i in range(0, 10):
            start_time = datetime.now()

            # print("Start Time:", start_time.strftime(time_format))
            runNeuralNetwork(n * n)

            end_time = datetime.now()
            # print("End Time:", end_time.strftime(time_format))
            execution_time = end_time - start_time
            # print("Execution Time:", execution_time)

            sum_time += execution_time.total_seconds()

        n = n * 2
        print("Average Execution Time (seconds):", sum_time/10)