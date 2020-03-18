import numpy as np
from Classes import NeuralNetwork
from Utils import *
import pygame
import matplotlib.pyplot as plt
from time import sleep

pygame.init()
screen = pygame.display.set_mode((1000,500))
pygame.display.set_caption("nn")
hiddenNUM = 8

inputs_raw = [1, 2, 3, 4, 5, 6]
normalize(inputs_raw,hiddenNUM)
'''
inputs = np.array([[0, 1, 1, 1, 0, 0, 1, 1],
                   [0, 1, 1, 1, 0, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 0, 1, 1, 1, 0],
                   [0, 1, 0, 0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0, 1, 0, 1],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 0, 0, 0, 0],
                   [1, 0, 1, 1, 0, 1, 0, 0]])'''

outputs_raw = [2, 3, 4, 5, 6, 7]
normalize(outputs_raw,hiddenNUM)
#outputs = [[1], [0], [0], [1], [0], [1], [1], [0], [1], [0]]

NN = NeuralNetwork(np.array(inputs_raw),np.array(outputs_raw))

clock = pygame.time.Clock()
cycles = 0
maxCycles = 1000
n_animations = 5

ex_raw = [8]
normalize(ex_raw,hiddenNUM)
example = np.array([ex_raw])

running = True
while running:
    screen.fill((0,0,0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if cycles <= n_animations*maxCycles:
        if cycles % n_animations == 0:
            NN.feed_forward()
            NN.backpropagation()
            screen.fill((0, 0, 0))
            drawNeuralNetwork(screen, [hiddenNUM, hiddenNUM, hiddenNUM], [1, 1, 0])
        else:
            screen.fill((0, 0, 0))
            l = [0, 0, 0]
            n = 3-cycles % n_animations
            if n != -1:
                l[n] = 1
            drawNeuralNetwork(screen, [hiddenNUM, hiddenNUM, hiddenNUM], l)
            NN.error_history.append(np.average(np.abs(NN.error)))
            NN.epoch_list.append(cycles)
        displayText(screen, str(cycles / n_animations), 9.3)
        cycles += 1
    else:
        displayText(screen, str(NN.showNeurons()), 2)
        print(NN.predict(example))
        displayText(screen,str(NN.predict(example)),1)

    pygame.display.flip()
    clock.tick(60)