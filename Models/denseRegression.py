from torch import nn
from torch.nn import ReLU

from NeuralLayers import KlLayers



class DenseRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, is_cuda = False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc1 = KlLayers.KlLayers(input_dim, 300, is_cuda)
        self.fc2 = KlLayers.KlLayers(300, 100, is_cuda)
        self.fc3 = KlLayers.KlLayers(100, output_dim, is_cuda)

        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3]
        self.reluList = [self.relu1, self.relu2]
        self.kl_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_layers:
            KLD += layer.kl_divergence()
        return KLD  