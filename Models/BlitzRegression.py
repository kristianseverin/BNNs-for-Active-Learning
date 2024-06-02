import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianRegressor, self).__init__()
        self.fc1 = BayesianLinear(input_dim, 512)
        self.fc2 = BayesianLinear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

        