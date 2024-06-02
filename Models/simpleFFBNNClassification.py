import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU
import torch.nn.functional as F
from NeuralLayers import KlLayers
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator
torch.manual_seed(43)

@variational_estimator
class SimpleFFBNNClassification(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleFFBNNClassification, self).__init__()
        self.fc1 = KlLayers.KlLayers(input_dim, 10)
        self.fc2 = KlLayers.KlLayers(10, 20)
        self.fc3 = KlLayers.KlLayers(20, output_dim)

        self.kl_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)
        x = self.fc3(x)
        return x
    
    def kl_divergence(self):
        kl = 0
        for layer in self.kl_layers:
            kl += layer.kl_divergence()
        return kl
    