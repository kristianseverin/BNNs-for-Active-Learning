import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU
import torch.nn.functional as F
from NeuralLayers import KlLayers
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

@variational_estimator
class SimpleFFBNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleFFBNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()


        self.fc1 = KlLayers.KlLayers(input_dim, 10)
        self.fc2 = KlLayers.KlLayers(10, 20)
        self.fc3 = KlLayers.KlLayers(20, output_dim)


        self.layers =[self.fc1, self.relu_1, self.fc2, self.relu_2, self.fc3]
        self.relu_layers = [self.relu_1, self.relu_2]
        self.kl_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):

        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        
        for layer in self.layers:
            x = layer(x)
        return x
    
    def kl_divergence(self):
        kl = 0
        for layer in self.kl_layers:
            kl += layer.kl_divergence()
        return kl
    