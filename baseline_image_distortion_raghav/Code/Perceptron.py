# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:55:00 2019

@author: ramacr1
"""

'''
	name: Perceptron.py
	purpose: Code up a perceptron
	date: March 2, 2019
'''

import math
from copy import deepcopy

class Perceptron:
	def __init__(self, weights, bias):
		# print("__init__ called!")
		self.weights = []
		self.bias = 0
		self.__check_weights(deepcopy(weights))
		self.delta_weights = [0] * self.wlines
		self.__check_bias(deepcopy(bias))
		self.delta_bias = 0
		self.activity = 0
		self.activation = 0
		self.delta = 0
		self.output = 0
		self.eta = 0
	# calculate activity value
	def calc_activity(self, inputs):
		# print("calc_activity() called!")
		try:
			self.__check_inputs(deepcopy(inputs))
			# print(self.inputs)
			if self.error == 1:
				raise AttributeError("1 or more variables not set properly.")
			else:
				# reset activity value
				self.activity = 0
				# compute activity value using A_j = sum_(w*x)+bias
				for i in range(self.wlines):
					self.activity += self.weights[i]*self.inputs[i]
				self.activity += self.bias*1
		except:
			raise
	# calculate activation value
	def calc_activation(self, activity=None):
		# print("calc_activation() called!")
		try:
			if self.error == 1:
				raise AttributeError("1 or more variables not set properly.")
			elif activity is not None:
				self.activity = float(activity)
			else:
				self.activity = self.activity
			# compute activation using sigmoid function y_j = 1/(1+e^(-A_j))
			# had to handle overflow issue with activity being too negative
			# 1/(1+e^(-x)) = e^x/(e^x+1)=1-1/(1+e^x)
			if self.activity < 0:
				self.activation = 1-1/(1+math.exp(self.activity))
			else:
				self.activation = 1/(1+math.exp(-1*self.activity))
		except:
			raise
	# set delta
	def set_delta(self, e):
		# print("set_delta() called!")
		try:
			self.__check_error(deepcopy(e))
			if self.error == 1:
				raise AttributeError("1 or more variables not set properly.")
			else:
				# compute delta value
				self.delta = e * (1 - self.activation) * self.activation
		except:
			raise
	# set change in weights
	def set_delta_weights(self, eta):
		# print("set_delta_weights() called!")
		try:
			self.__check_eta(deepcopy(eta))
			if self.error == 1:
				raise AttributeError("1 or more variables not set properly.")
			else:
				# compute change in weights eta*(-e_j[1-y_j]y_j)
				for i in range(self.wlines):
					self.delta_weights[i] = -1 * eta * self.delta * self.inputs[i]
				self.delta_bias = -1 * eta * self.delta * 1
		except:
			raise
	# update weights
	def update_weights(self):
		# print("update_weights() called!")	
		try:
			if self.error == 1:
				raise AttributeError("1 or more variables not set properly.")
			else:
				# subtract delta weights*x_i from current weights
				for i in range(self.wlines):
					self.weights[i] -= self.delta_weights[i]
				self.bias -= self.delta_bias
		except:
			raise
	# check weight type, values
	def __check_weights(self, weights):
		self.error = 1
		try:
			wlines = len(weights)
			if wlines > 0:
				for i in range(wlines):
					weights[i] = float(weights[i]) 
				self.error = 0
				self.weights = weights
				self.wlines = wlines
			else:
				raise ValueError("Length of list is 0.")
		except:
			raise
	# check bias type
	def __check_bias(self, bias):
		self.error = 1
		try:
			self.bias = float(bias)
			self.error = 0
		except:
			raise
	# check inputs type, values
	def __check_inputs(self, inputs):
		self.error = 1
		try:
			ilines = len(inputs)
			if ilines == 0:
				raise ValueError("Length of list is 0.")
			elif ilines != self.wlines:
				raise ValueError("Length of list is not equal to length of weights ("+str(len(self.weights))+")")
			else:
				for i in range(ilines):
					inputs[i] = float(inputs[i])
				self.inputs = inputs
				self.error = 0
		except:
			raise
	# check error type
	def __check_error(self, e):
		self.error = 1
		try:
			self.error = 0
		except:
			raise
	# check eta type
	def __check_eta(self, eta):
		self.error = 1
		try:
			self.error = 0
		except:
			raise