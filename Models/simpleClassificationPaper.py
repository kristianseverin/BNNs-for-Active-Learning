import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU
import torch.nn.functional as F
from NeuralLayers import PaperLayers
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator
torch.manual_seed(43)

@variational_estimator
class SimpleFFBNNClassificationPaper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = PaperLayers.LinearGroupNJ(input_dim, 10)
        self.fc2 = PaperLayers.LinearGroupNJ(10, 20)
        self.fc3 = PaperLayers.LinearGroupNJ(20, output_dim)

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
    