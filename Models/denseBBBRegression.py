from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from blitz.losses.kl_divergence import kl_divergence_from_nn
import torch.nn as nn

@variational_estimator
class DenseBBBRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.i_dim = input_dim
        self.o_dim = output_dim
        prior_sigma_1 = 1.0
        prior_sigma_2 = 0.0025
        prior_pi = 0.5
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.fc1 = BayesianLinear(input_dim, 300, prior_sigma_1=prior_sigma_1, prior_sigma_2 = prior_sigma_2, prior_pi=prior_pi) # look into posterior_rho_init
        self.fc2 = BayesianLinear(300, 100, prior_sigma_1=prior_sigma_1, prior_sigma_2 = prior_sigma_2, prior_pi=prior_pi)
        self.fc3 = BayesianLinear(100, output_dim, prior_sigma_1=prior_sigma_1, prior_sigma_2 = prior_sigma_2, prior_pi=prior_pi)

        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3]
        self.reluList = [self.relu1, self.relu2]
        self.kl_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def kl_divergence(self):
        kl = 0
        for layer in self.kl_layers:
            kl += kl_divergence_from_nn(layer)
        return kl


        


