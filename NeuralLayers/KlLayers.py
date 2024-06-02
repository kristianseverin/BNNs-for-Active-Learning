import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules import utils
import math

def reparameterize(mu, logvar, cuda = False, sample = True):
    """ Function for reparameterization. .
        This is based of the paper "INSERT PAPER NAME HERE" by INSERT AUTHORS HERE (INSERT YEAR HERE)"
    """
    if sample:
        std = logvar.mul(0.5).exp_()
        if cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        
        eps = torch.autograd.Variable(eps)
        print(f'this is mu: {mu}, this is std: {std}, this is eps: {eps}')
        return mu + std * eps
    else:
        return 

class KlLayers(Module):
    """ Class for KL divergence layers. This is an implementation of Fully Connected Group Normal-Jeffrey's layer.
        This is based of the paper "Efficacy of Bayesian Neural Networks in Active Learning" by Rakish & Jain (2017)
    """

    def __init__(self, in_features, out_features, cuda = False, initial_weights = None, initial_bias = None, clip_variance = None):
        super(KlLayers, self).__init__()
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features
        self.clip_variance = clip_variance
        self.dropout_mu = nn.Parameter(torch.Tensor(in_features)) # mean of the Gaussian distribution used for sampling dropout masks
        self.dropout_logvar = nn.Parameter(torch.Tensor(in_features)) # log variance of the Gaussian distribution used for sampling dropout masks
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features)) # mean of the Gaussian distribution used for sampling weights
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features)) # log variance of the Gaussian distribution used for sampling weights
        self.bias_mu = nn.Parameter(torch.Tensor(out_features)) # mean of the Gaussian distribution used for sampling biases
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features)) # log variance of the Gaussian distribution used for sampling biases

        # initialize parameters
        self.reset_parameters(initial_weights, initial_bias)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.epsilon = 1e-8 # epsilon value used for numerical stability


    def reset_parameters(self, initial_weights, initial_bias):
        """ Function for initializing the parameters of the layer. """
        stdv = 1. / math.sqrt(self.weight_mu.size(1))

        self.dropout_mu.data.normal_(1, 1e-2)

        if initial_weights is not None:
            self.weight_mu.data = torch.Tensor(initial_weights)
        else:
            self.weight_mu.data.normal_(0, stdv)

        if initial_bias is not None:
            self.bias_mu.data = torch.Tensor(initial_bias)
        else:
            self.bias_mu.data.fill_(0)

        self.dropout_logvar.data.normal_(-9, 1e-2)  # have to figure out why -9, 1e-2  (maybe -9 = log(1e-4))
        self.weight_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def clip_variances(self):
        """ Function for clipping the variances of the layer. """
        if self.clip_variance:
            self.weight_logvar.data.clamp_(max=math.log(self.clip_variance))
            self.bias_logvar.data.clamp_(max=math.log(self.clip_variance))

    def get_log_dropout_rates(self):
        log_alpha = self.dropout_logvar - torch.log(self.dropout_mu.pow(2) + self.epsilon)
        return log_alpha

    def posterior_parameters(self):
        """ Function for getting the posterior parameters of the layer. """
        weight_variance, dropout_variance = self.weight_logvar.exp(), self.dropout_logvar.exp()
        self.posterior_weight_variance = self.dropout_mu.pow(2) * weight_variance + dropout_variance * self.weight_mu.pow(2) * weight_variance
        self.posterior_weight_mean = self.weight_mu * self.dropout_mu
        return self.posterior_weight_mean, self.posterior_weight_variance

    def forward(self, x):
        """ Function for forward pass. """
        #self.posterior_parameters()
        #return F.linear(x, self.posterior_weight_mean, self.bias_mu)

        batch_size = x.size()[0]
        print(f' this is batch_size: {batch_size}')

        print(f'this is self.dropout_mu: {self.dropout_mu}, this is self.dropout_logvar: {self.dropout_logvar}')
        print(f'these are the shapes: {self.dropout_mu.repeat(batch_size, 1).shape}, {self.dropout_logvar.repeat(batch_size, 1).shape}')

        z = reparameterize(self.dropout_mu.repeat(batch_size, 1), self.dropout_logvar.repeat(batch_size, 1), sample = self.training, cuda = self.cuda)
        #print(f'this is z: {z}, this is x: {x}')

        # local reparameterization trick
        xz = x * z
        mu_activation = F.linear(xz, self.weight_mu, self.bias_mu)
        var_activation = F.linear(xz.pow(2), self.weight_logvar.exp(), self.bias_logvar.exp())

        return reparameterize(mu_activation, var_activation.log(), sample = True, cuda = self.cuda)

    def kl_divergence(self):
        """ Function for calculating the KL divergence of the layer. """
        k1, k2, k3 = 0.63576, 1.87320, 1.48695 # approximations made by Molchanov et al. (2017)
        log_alpha = self.get_log_dropout_rates()
        KLD = -torch.sum(k1 * self.sigmoid(k2 + k3 * log_alpha) - 0.5 * self.softplus(-log_alpha) - k1)

        KLD_element = -0.5 *self.weight_logvar + 0.5 *(self.weight_logvar.exp() + self.weight_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)

        KLD_element = -0.5 *self.bias_logvar + 0.5 *(self.bias_logvar.exp() + self.bias_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)

        return KLD

    # for debugging purposes
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
