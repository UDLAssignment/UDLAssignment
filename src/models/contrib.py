"""
This module contains the less thoroughly tested implementations of VCL models.

In particular, the models in this module are defined in a different manner to the main
models in the models.vcl_nn module. The models in this module are defined in terms of
bayesian layers from the layers.variational module, which abstract the details of
online variational inference. This approach is in line with the standard style in which
PyTorch models are defined.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.variational import VariationalLayer, MeanFieldGaussianLinear, LowRankMeanFieldGaussianLinear, FlowScalingLinear, CompleteFlowScalingLinear, FlowMFScalingLinear
from models.deep_models import Encoder
from util.operations import kl_divergence, bernoulli_log_likelihood, normal_with_reparameterization

EPSILON = 1e-8  # Small value to avoid divide-by-zero and log(zero) problems

class VCL(nn.Module, ABC):
    """ Base class for all VCL models """
    def __init__(self, epsilon=EPSILON):
        super().__init__()
        self.epsilon = epsilon

    @abstractmethod
    def reset_for_new_task(self, head_idx):
        pass


class DiscriminativeVCL(VCL):
    """
    A Bayesian neural network which is updated using variational inference
    methods, and which is multi-headed at the output end. Suitable for
    continual learning of discriminative tasks.
    """

    def __init__(self, config, x_dim, writer=None):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = config.h_dim
        self.y_dim = config.n_classes
        self.n_heads = config.n_tasks if config.multiheaded else 1
        self.ipv = config.initial_posterior_var
        self.mc_sampling_n = config.mc_sampling_n
        self.device = config.device
        self.config = config

        shared_dims = [x_dim] + list(config.shared_h_dims) + [config.h_dim]

        if config.posterior_type == "meanfield":
            linear_class = MeanFieldGaussianLinear
        elif config.posterior_type == "meanfield_lowrankvar":
            linear_class = LowRankMeanFieldGaussianLinear
        elif config.posterior_type == "flow":
            linear_class = FlowScalingLinear
        elif config.posterior_type == "complete_flow":
            linear_class = CompleteFlowScalingLinear
        elif config.posterior_type == "flow_meanfield":
            linear_class = FlowMFScalingLinear

        # list of layers in shared network
        self.shared_layers = nn.ModuleList([
            linear_class(config, shared_dims[i], shared_dims[i + 1], self.ipv, EPSILON, writer=writer, name=f"shared_{i}") for i in
            range(len(shared_dims) - 1)
        ])
        # list of heads, each head is a list of layers
        self.heads = nn.ModuleList([
            linear_class(config, self.h_dim, self.y_dim, self.ipv, EPSILON, writer=writer, name=f"head_{i}") for i in range(self.n_heads)
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, head_idx, num_samples=1, sample_parameters=True, do_softmax=False, do_mean=False):
        # y_out = torch.zeros(size=(x.size()[0], self.y_dim)).to(self.device)
        outs = []

        # repeat forward pass n times to sample layer params multiple times
        for _ in range(num_samples):
            h = x
            # shared part
            for layer in self.shared_layers:
                h = F.relu(layer(h, sample_parameters=sample_parameters))

            # head
            h = self.heads[head_idx](h, sample_parameters=sample_parameters)

            if do_softmax:
                h = self.softmax(h)
            
            outs.append(h)

        #     y_out.add_(h)

        # y_out.div_(num_samples)
        if do_mean:
            outs = sum(outs) / len(outs)

        return outs
    
    def get_logs(self):
        logs = {}
        for i, layer in enumerate(self.shared_layers):
            logs.update({f"shared_{i}/" + k: v for k, v in layer.get_logs().items()})
        for i, layer in enumerate(self.heads):
            logs.update({f"head_{i}/" + k: v for k, v in layer.get_logs().items()})
        return logs
    
    def _log_prob(self, x, y, head, num_samples, sample_parameters=True):
        outputs = self(x, head, num_samples, sample_parameters=sample_parameters)
        return - nn.CrossEntropyLoss()(torch.cat(outputs), y.repeat(num_samples).view(-1))

    def vcl_loss(self, x, y, head_idx, task_size):
        kl,log_prob = self._kl_divergence(head_idx) / task_size, - self._log_prob(x, y, head_idx, num_samples=self.config.num_train_samples, sample_parameters=True)
        return kl, log_prob
        # kl,log_prob = self._kl_divergence(head_idx), - self._log_prob(x, y, head_idx, num_samples=self.config.num_train_samples, sample_parameters=True)
        # if self.config.posterior_type == "meanfield":
        #     kl = kl / task_size
        # return kl, log_prob

    def point_estimate_loss(self, x, y, head_idx):
        return torch.nn.CrossEntropyLoss()(self(x, head_idx, sample_parameters=False)[0], y)

    def prediction(self, x, task):
        """ Returns an integer between 0 and self.out_size """
        return torch.argmax(self(x, task, num_samples=self.mc_sampling_n, do_softmax=True, do_mean=True), dim=1)
    
    def load_old_model(self, old_model):
        # shared_layers
        for old_prior, old_posterior, new_layer in zip(old_model.prior[0][0], old_model.posterior[0][0], self.shared_layers):
            new_layer.prior_W_means.data = old_prior.T
            new_layer.posterior_W_means.data = old_posterior.T
        
        for old_prior, old_posterior, new_layer in zip(old_model.prior[0][1], old_model.posterior[0][1], self.shared_layers):
            new_layer.prior_W_log_vars.data = old_prior.T
            new_layer.posterior_W_log_vars.data = old_posterior.T

        for old_prior, old_posterior, new_layer in zip(old_model.prior[1][0], old_model.posterior[1][0], self.shared_layers):
            new_layer.prior_b_means.data = old_prior
            new_layer.posterior_b_means.data = old_posterior
        
        for old_prior, old_posterior, new_layer in zip(old_model.prior[1][1], old_model.posterior[1][1], self.shared_layers):
            new_layer.prior_b_log_vars.data = old_prior
            new_layer.posterior_b_log_vars.data = old_posterior

        # heads
        for old_prior, old_posterior, new_layer in zip(old_model.head_prior[0][0], old_model.head_posterior[0][0], self.heads):
            new_layer.prior_W_means.data = old_prior.T
            new_layer.posterior_W_means.data = old_posterior.T
        
        for old_prior, old_posterior, new_layer in zip(old_model.head_prior[0][1], old_model.head_posterior[0][1], self.heads):
            new_layer.prior_W_log_vars.data = old_prior.T
            new_layer.posterior_W_log_vars.data = old_posterior.T

        for old_prior, old_posterior, new_layer in zip(old_model.head_prior[1][0], old_model.head_posterior[1][0], self.heads):
            new_layer.prior_b_means.data = old_prior
            new_layer.posterior_b_means.data = old_posterior
        
        for old_prior, old_posterior, new_layer in zip(old_model.head_prior[1][1], old_model.head_posterior[1][1], self.heads):
            new_layer.prior_b_log_vars.data = old_prior
            new_layer.posterior_b_log_vars.data = old_posterior

    def reset_for_new_task(self, head_idx):
        for layer in self.shared_layers:
            if isinstance(layer, VariationalLayer):
                layer.reset_for_next_task()

        if isinstance(self.heads[head_idx], VariationalLayer):
            self.heads[head_idx].reset_for_next_task()
    
    def start_variational_phase(self):
        for layer in self.shared_layers:
            if isinstance(layer, VariationalLayer):
                layer.start_variational_phase()
            
        for layer in self.heads:
            if isinstance(layer, VariationalLayer):
                layer.start_variational_phase()


    def log_stats(self, logger) -> (list, dict):
        
        model_statistics = {
            'average_w_mean': 0,
            'average_b_mean': 0,
            'average_w_var': 0,
            'average_b_var': 0
        }

        n_layers = 0
        for layer in self.shared_layers:
            n_layers += 1
            layer_statistics.append(layer.get_statistics())
            model_statistics['average_w_mean'] += layer_statistics[-1]['average_w_mean']
            model_statistics['average_b_mean'] += layer_statistics[-1]['average_b_mean']
            model_statistics['average_w_var'] += layer_statistics[-1]['average_w_var']
            model_statistics['average_b_var'] += layer_statistics[-1]['average_b_var']

        for head in self.heads:
            n_layers += 1
            layer_statistics.append(head.get_statistics())
            model_statistics['average_w_mean'] += layer_statistics[-1]['average_w_mean']
            model_statistics['average_b_mean'] += layer_statistics[-1]['average_b_mean']
            model_statistics['average_w_var'] += layer_statistics[-1]['average_w_var']
            model_statistics['average_b_var'] += layer_statistics[-1]['average_b_var']

        # todo averaging averages like this is actually incorrect (assumes equal num of params in each layer)
        model_statistics['average_w_mean'] /= n_layers
        model_statistics['average_b_mean'] /= n_layers
        model_statistics['average_w_var'] /= n_layers
        model_statistics['average_b_var'] /= n_layers

        return layer_statistics, model_statistics

    def _kl_divergence(self, head_idx) -> torch.Tensor:
        kl_divergence = torch.zeros(1, requires_grad=False).to(self.device)

        # kl divergence is equal to sum of parameter-wise divergences since
        # distribution is diagonal multivariate normal (parameters are independent)
        for layer in self.shared_layers:
            kl_divergence = torch.add(kl_divergence, layer.kl_divergence())

        kl_divergence = torch.add(kl_divergence, self.heads[head_idx].kl_divergence())
    
        return kl_divergence