from random import uniform
import numpy as np
import pygame
from Utils import *
from time import sleep
'''
class NeuralNetwork2:
    def __init__(self, inputs, outputs, hidden):
        self.inputs = inputs
        self.outputs = outputs
        self.hiddenNUM = hidden

        hiddenNeurons = [[uniform(0,1)] for a in range(self.hiddenNUM)]
        self.weights = np.array(hiddenNeurons)

        self.error_history = []
        self.epoch_list = []

    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    def backpropagation(self):
        self.error = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, True)
        self.weights += np.dot(self.inputs.T, delta)

    def train(self, epochs=1):
        for epoch in range(epochs):
            self.feed_forward()

            self.backpropagation()
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

    def showNeurons(self):
        return self.weights
'''

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.times = len(inputs[0])

        weigth_raw = []
        for i in range(self.times):
            weigth_raw.append([])
            while len(weigth_raw[i]) <= (self.times - 1):
                weigth_raw[i].append([uniform(0, 1)])

        self.weights = np.array(weigth_raw)
        self.error_history = []
        self.epoch_list = []

    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def feed_forward(self):
        self.hidden = []
        for i in range(self.times):
            self.hidden.append(self.sigmoid(np.dot(self.inputs, self.weights[i])))

    def backpropagation(self):
        for i in range(self.times):
            outputs_transpose = self.outputs.T[i]

            output_raw = []
            for e in outputs_transpose:
                output_raw.append([e])

            self.error = np.array(output_raw) - self.hidden[i]

            delta = self.error * self.sigmoid(self.hidden[i], deriv=True)
            self.weights[i] += np.dot(self.inputs.T, delta)

    def train(self, epochs=100000):
        for epoch in range(epochs):
            self.feed_forward()

            self.backpropagation()
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    def predict(self, new_input):
        predictions = []
        for i in range(self.times):
            prediction = self.sigmoid(np.dot(new_input, self.weights[i]))
            predictions.append(prediction)
        return predictions

    def showNeurons(self):
        return self.weights