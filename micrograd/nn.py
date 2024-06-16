import numpy as np
from micrograd.engine import Value


class Module():
    def __init__(self):
        pass
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def update_parameters(self, lr):
        for p in self.parameters():
            p.data -= p.grad*lr


class Neuron(Module):
    def __init__(self, nin):
        """ nin is a scalar which denotes the dimensionality of the vector (i.e num features)"""
        self.weights = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(0)

    def forward(self, x):
        output = sum([x_i*w_i for x_i, w_i in zip(x, self.weights)], self.bias)
        return output
    
    def parameters(self):
       return self.weights + [self.bias]


class Layer(Module):
    def __init__(self, nin, nout):
        """ nout is a scalar which denotes the number output neurons for the layer """
        self.layer = [Neuron(nin) for _ in range(nout)]

    def forward(self, x):
        output = [neuron.forward(x) for neuron in self.layer]
        return output

    def parameters(self):
        parameters = [p for neuron in self.layer for p in neuron.parameters()]
        return parameters

class MLP(Module):
    def __init__(self, nin, nouts):
        """ nouts is a list which denotes the number of output neurons for each layer; where num_layers == len(list) """
        self.nouts = [nin] + nouts
        self.layers = [Layer(self.nouts[idx], self.nouts[idx+1]) for idx in range(len(self.nouts) - 1)]

    def forward(self, x):
        for layer_idx in range(len(self.nouts) - 1):
            x = self.layers[layer_idx].forward(x)
            if layer_idx != (len(self.nouts) - 2): #Ensures output layer does not use ReLU non-linearity, only used in intermediate layers
                x = [x_i.relu() for x_i in x]
        output = x
        return output
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend([layer for layer in layer.parameters()])
        return parameters
