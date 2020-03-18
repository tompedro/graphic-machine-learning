import pygame

colorActive = 66,133,244


def decimal_to_bin(n, l):
    if n == 1:
        l.append(1)
        return
    else:
        l.append(round(n%2))
        return decimal_to_bin((n-(n%2)) / 2,l)


def displayText(screen, string, y, x=0):
    text = pygame.font.SysFont('Arial', 28).render(string, False, (255,255,255))
    screen.blit(text, (x, y * 32))


def foundPosNeuron(n_layer,n_neuron):
    x, y = 0, 400 - (n_neuron - 1) * 50
    if n_layer == 0:
        x = 300
    elif n_layer == 1:
        x = 500
    else:
        x = 700
    return [x,y]


def drawNeuron(screen,n_layer,n_neuron,state):
    pos = foundPosNeuron(n_layer,n_neuron)
    if state:
        color = colorActive
    else:
        color = 255,255,255

    pygame.draw.circle(screen,color,pos,10)
    return pos

def drawLink(screen,pos1,pos2,state):
    if state:
        color = colorActive
    else:
        color = 255,255,255
    pygame.draw.line(screen,color,pos1,pos2)

def drawNeuralNetwork(screen,n_neurons,layer):
    positions = [[foundPosNeuron(j,i+1) for i in range(n_neurons[j])] for j in range(3)]

    for coord1 in positions[1]:
        for coord in positions[0]:
            drawLink(screen, coord1, coord, layer[0])
        for coord2 in positions[2]:
            drawLink(screen, coord1, coord2, layer[2])

    for i in range(n_neurons[2]):
        drawNeuron(screen, 2, i + 1, layer[2])
    for i in range(n_neurons[1]):
        drawNeuron(screen, 1, i + 1, layer[1])
    for i in range(n_neurons[0]):
        drawNeuron(screen, 0, i + 1, layer[0])


def normalize(l,maxN):
    for i in range(len(l)):
        m = []
        decimal_to_bin(l[i], m)
        m.extend([0 for x in range(maxN - len(m))])
        m.reverse()
        l[i] = m