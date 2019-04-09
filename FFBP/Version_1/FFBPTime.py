'''
    name: FFBPTiming.py
    purpose: Test ffbp timing for different nxn input sizes
    date: 04/06/2019
'''

import matplotlib.image as mpimg
import matplotlib.pyplot as mpplot
import numpy as np
import math as math
from Layer import Layer
from time import clock

class FFBPTiming:
    def __init__(self):
        # print("__init__ called")
        self.npImg = []
    def readPNG(self):
        # print("readPNG called")
        img = mpimg.imread('../Images/007.png')
        self.npImg = np.array(img)
        self.npImg = self.npImg.flatten()
    def runNN(self, l, w, ntests):
        # print("runNN called")
        self.l1_input = self.npImg[:l*w,].tolist()
        self.l1_weights = np.random.randn(2*l*w/3, l*w).tolist()
        self.l1_bias = np.zeros(2*l*w/3).tolist()
        self.l1 = Layer(0, 2*l*w/3, self.l1_weights, self.l1_bias)
        self.l2_weights = np.random.randn(1, 2*l*w/3).tolist()
        self.l2_bias = np.zeros(1).tolist()
        self.l2 = Layer(1, 1, self.l2_weights, self.l2_bias)
        self.times = []
        self.start_time = 0
        self.end_time = 0
        for t in range(ntests):
            self.start_time = clock()
            self.l2.get_layer_output_vector(self.l1.get_layer_output_vector(self.l1_input))
            self.l2.get_error_vector([1])
            self.l2.set_output_layer_delta_values()
            self.l1.set_hidden_layer_delta_values(self.l2)
            self.l2.calc_layer_delta_weights(1.0)
            self.l1.calc_layer_delta_weights(1.0)
            self.l2.update_layer_weights()
            self.l1.update_layer_weights()
            self.end_time = clock()
            self.times.append(self.end_time - self.start_time)
        return self.times
    def plotNNBatch(self):
        # print("runNNBatch called")
        data = []
        mindim = 2
        maxdim = 7
        dim = []
        for i in range(mindim, maxdim):
            dim.append(int(math.pow(2,i)))
            print(dim[i-mindim])
        iter = 10
        for i in dim:
            data.append(sum(self.runNN(i,i,iter))/iter)
        mpplot.scatter(dim, data)
        mpplot.xlabel("NxN data region")
        mpplot.ylabel("Time(s)")
        mpplot.show()

if __name__=='__main__':
    f = FFBPTiming()
    f.readPNG()
    f.plotNNBatch()

