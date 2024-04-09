from abc import ABC, abstractmethod
import math
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import copy
from nflows import transforms, distributions, flows
from nflows.nn import nets as nets
import einops

class VariationalLayer(torch.nn.Module, ABC):
    """
    Base class for any type of neural network layer that uses variational inference. The
    defining aspect of such a layer are:

    1. The PyTorch forward() method should allow the parametric specification of whether
    the forward pass should be carried out with parameters sampled from the posterior
    parameter distribution, or using the distribution parameters directly as if though
    they were standard neural network parameters.

    2. Must provide a method for resetting the parameter distributions for the next task
    in the online (multi-task) variational inference setting.

    3. Must provide a method for computing the KL divergence between the layer's parameter
    posterior distribution and parameter prior distribution.
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    @abstractmethod
    def forward(self, x, sample_parameters=True):
        pass

    @abstractmethod
    def reset_for_next_task(self):
        pass

    @abstractmethod
    def kl_divergence(self) -> torch.Tensor:
        pass

    def start_variational_phase(self):
        return


class MeanFieldGaussianLinear(VariationalLayer):
    """
    A linear transformation on incoming data of the form :math:`y = w^T x + b`,
    where the weights w and the biases b are distributions of parameters
    rather than point estimates. The layer has a prior distribution over its
    parameters, as well as an approximate posterior distribution.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    def __init__(self, config, in_features, out_features, initial_posterior_variance=1e-3, epsilon=1e-8, name="", writer=None):
        super().__init__(epsilon)
        self.in_features = in_features
        self.out_features = out_features
        self.ipv = initial_posterior_variance

        # priors are not optimizable parameters - all means and log-variances are zero
        self.register_buffer('prior_W_means', torch.zeros(out_features, in_features))
        self.register_buffer('prior_W_log_vars', torch.zeros(out_features, in_features))
        self.register_buffer('prior_b_means', torch.zeros(out_features))
        self.register_buffer('prior_b_log_vars', torch.zeros(out_features))

        self.posterior_W_means = Parameter(torch.empty_like(self._buffers['prior_W_means'], requires_grad=True))
        self.posterior_b_means = Parameter(torch.empty_like(self._buffers['prior_b_means'], requires_grad=True))
        self.posterior_W_log_vars = Parameter(torch.empty_like(self._buffers['prior_W_log_vars'], requires_grad=True))
        self.posterior_b_log_vars = Parameter(torch.empty_like(self._buffers['prior_b_log_vars'], requires_grad=True))

        self._initialize_posteriors()

    def forward(self, x, sample_parameters=True):
        """ Produces module output on an input. """
        if sample_parameters:
            w, b = self._sample_parameters()
            return F.linear(x, w, b)
        else:
            return F.linear(x, self.posterior_W_means, self.posterior_b_means)

    def reset_for_next_task(self):
        """ Overwrites the current prior with the current posterior. """
        self._buffers['prior_W_means'].data.copy_(self.posterior_W_means.data)
        self._buffers['prior_W_log_vars'].data.copy_(self.posterior_W_log_vars.data)
        self._buffers['prior_b_means'].data.copy_(self.posterior_b_means.data)
        self._buffers['prior_b_log_vars'].data.copy_(self.posterior_b_log_vars.data)

    def kl_divergence(self) -> torch.Tensor:
        """ Returns KL(posterior, prior) for the parameters of this layer. """
        # obtain flattened means, log variances, and variances of the prior distribution
        prior_means = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_means'], (-1,)),
             torch.reshape(self._buffers['prior_b_means'], (-1,)))),
            requires_grad=False
        )
        prior_log_vars = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_log_vars'], (-1,)),
             torch.reshape(self._buffers['prior_b_log_vars'], (-1,)))),
            requires_grad=False
        )
        prior_vars = torch.exp(prior_log_vars)

        # obtain flattened means, log variances, and variances of the approximate posterior distribution
        posterior_means = torch.cat(
            (torch.reshape(self.posterior_W_means, (-1,)),
             torch.reshape(self.posterior_b_means, (-1,))),
        )
        posterior_log_vars = torch.cat(
            (torch.reshape(self.posterior_W_log_vars, (-1,)),
             torch.reshape(self.posterior_b_log_vars, (-1,))),
        )
        posterior_vars = torch.exp(posterior_log_vars)

        # compute kl divergence (this computation is valid for multivariate diagonal Gaussians)
        kl_elementwise = posterior_vars / (prior_vars + self.epsilon) + \
                         torch.pow(prior_means - posterior_means, 2) / (prior_vars + self.epsilon) - \
                         1 + prior_log_vars - posterior_log_vars

        return 0.5 * kl_elementwise.sum()

    def get_statistics(self) -> dict:
        statistics = {
            'average_w_mean': torch.mean(self.posterior_W_means),
            'average_b_mean': torch.mean(self.posterior_b_means),
            'average_w_var': torch.mean(torch.exp(self.posterior_W_log_vars)),
            'average_b_var': torch.mean(torch.exp(self.posterior_b_log_vars))
        }

        return statistics

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    def get_logs(self):
        return {}

    def _sample_parameters(self):
        # obtained sampled weights and biases using local reparameterization trick
        w_epsilons = torch.randn_like(self.posterior_W_means)
        b_epsilons = torch.randn_like(self.posterior_b_means)

        w = self.posterior_W_means + torch.mul(w_epsilons, torch.exp(0.5 * self.posterior_W_log_vars))
        b = self.posterior_b_means + torch.mul(b_epsilons, torch.exp(0.5 * self.posterior_b_log_vars))

        return w, b

    def _initialize_posteriors(self):
        # posteriors on the other hand are optimizable parameters - means are normally distributed, log_vars
        # have some small initial value

        torch.nn.init.normal_(self.posterior_W_means, mean=0, std=0.1)
        torch.nn.init.uniform_(self.posterior_b_means, -0.1, 0.1)
        torch.nn.init.constant_(self.posterior_W_log_vars, math.log(self.ipv))
        torch.nn.init.constant_(self.posterior_b_log_vars, math.log(self.ipv))

class LowRankPosterior(nn.Module):
    def __init__(self, shape, rank):
        super().__init__()
        self.shape = shape
        in_dim, out_dim = shape
        self.A = nn.Parameter(torch.ones(in_dim, rank))
        self.B = nn.Parameter(torch.full([out_dim, rank], math.log(1e-6)))
    
    def forward(self):
        return self.A @ self.B.T

class LowRankMeanFieldGaussianLinear(MeanFieldGaussianLinear):
    """
    A linear transformation on incoming data of the form :math:`y = w^T x + b`,
    where the weights w and the biases b are distributions of parameters
    rather than point estimates. The layer has a prior distribution over its
    parameters, as well as an approximate posterior distribution.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    def __init__(self, config, in_features, out_features, initial_posterior_variance=1e-3, epsilon=1e-8, name="", writer=None):
        super().__init__(config, in_features, out_features, initial_posterior_variance=initial_posterior_variance, epsilon=epsilon)
        self.in_features = in_features
        self.out_features = out_features
        self.ipv = initial_posterior_variance

        # priors are not optimizable parameters - all means and log-variances are zero
        self.register_buffer('prior_W_means', torch.zeros(out_features, in_features))
        self.register_buffer('prior_W_log_vars', torch.zeros(out_features, in_features))
        self.register_buffer('prior_b_means', torch.zeros(out_features))
        self.register_buffer('prior_b_log_vars', torch.zeros(out_features))

        self.posterior_W_means = Parameter(torch.empty_like(self._buffers['prior_W_means'], requires_grad=True))
        self.posterior_b_means = Parameter(torch.empty_like(self._buffers['prior_b_means'], requires_grad=True))
        self.posterior_W_log_var_net = LowRankPosterior(self._buffers['prior_W_log_vars'].shape, config.postvar_weight_rank)
        self.posterior_b_log_vars = Parameter(torch.empty_like(self._buffers['prior_b_log_vars'], requires_grad=True))

        self._initialize_posteriors()
    
    @property
    def posterior_W_log_vars(self):
        return self.posterior_W_log_var_net()

class StandardGaussianPrior():
    def __init__(self, center=None):
        self.center = center

    def add_center(self, center):
        self.center = center

    def log_prob(self, x):
        "Returns the log prob of x according to a standard normal distribution."
        if self.center is None: 
            center = torch.zeros_like(x)
        else:
            center = self.center
        return -0.5 * ((x-center) ** 2 + math.log(2 * math.pi))

class InvertibleNetwork(nn.Module):
    def __init__(self, config, out_features):
        super().__init__()

        mask = torch.ones(out_features)
        mask[::2] = -1

        def create_resnet(in_feat, out_feat):
            return nets.ResidualNet(
                in_feat,
                out_feat,
                out_features
            )

        layers = []
        for _ in range(2*config.num_flow_layers):
            layers.append(
                transforms.AffineCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=create_resnet
                )
            )
            mask *= -1

        # Define an invertible transformation.
        transform = transforms.CompositeTransform(layers)

        base_distribution = distributions.StandardNormal(shape=[out_features])
        self.flow = flows.Flow(transform=transform, distribution=base_distribution)

    def forward(self, n_samples=1):
        return self.flow.sample(n_samples)
    
    def log_prob(self, x):
        return self.flow.log_prob(x)

class FlowScalingLinear(VariationalLayer):

    def __init__(self, config, in_features, out_features, initial_posterior_variance=1e-3, epsilon=1e-8, name="", writer=None):
        super().__init__(epsilon)
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.ipv = initial_posterior_variance
        self.name = name
        self.writer = writer
        self.step = 0

        # priors are not optimizable parameters - all means and log-variances are zero
        self.base_weights = nn.Linear(in_features, out_features, bias=True)
        self.prior_base_weights = None
        self.prior = StandardGaussianPrior()
        self.posterior = InvertibleNetwork(config, out_features)

    def forward(self, x, sample_parameters=True):
        """ Produces module output on an input. """
        
        if sample_parameters:
            out_scales = self.posterior()[0]
            w = (self.base_weights.weight / torch.norm(self.base_weights.weight, p=2, dim=1)[:,None]) * out_scales[:,None]
            return F.linear(x, w, self.base_weights.bias)
        else:
            w = self.base_weights.weight / torch.norm(self.base_weights.weight, p=2, dim=1)[:,None]
            return F.linear(x, w, self.base_weights.bias)

    def reset_for_next_task(self):
        """ Overwrites the current prior with the current posterior. """
        self.prior = copy.deepcopy(self.posterior)
        self.prior_base_weights = copy.deepcopy(self.base_weights)
        self.step = 0
    
    def get_logs(self):
        return self.kl_divergence(return_log=True)

    def kl_divergence(self, return_log=False) -> torch.Tensor:
        """ Monte carlo estimate of the KL divergence for this layer """
        # obtain flattened means, log variances, and variances of the prior distribution
        kl_estimate = 0
        sample = self.posterior(self.config.num_kl_estimate_samples)
        prior_log_probs = self.prior.log_prob(sample)
        if len(prior_log_probs.shape) == 1:
            prior_log_probs = prior_log_probs.unsqueeze(1)
        kl_estimate -= prior_log_probs.sum(1).mean()
        kl_estimate += self.posterior.log_prob(sample).mean()

        kl_estimate_base_weight = 0
        kl_estimate_base_bias = 0
        if self.prior_base_weights is not None:
            kl_estimate_base_weight = (self.prior_base_weights.weight - self.base_weights.weight).pow(2).sum()
            kl_estimate_base_bias = (self.prior_base_weights.bias - self.base_weights.bias).pow(2).sum()
        
        if return_log:
            return {
                "kl": kl_estimate / self.config.num_kl_estimate_samples * self.config.kl_loss_coef, 
                "kl_base_weight":kl_estimate_base_weight* self.config.kl_base_loss_coef, 
                "kl_base_bias":kl_estimate_base_bias* self.config.kl_base_loss_coef,
                "abs_mean_weight": torch.abs(self.base_weights.weight).mean(),
                "abs_mean_bias": torch.abs(self.base_weights.bias).mean()
            }

        return kl_estimate / self.config.num_kl_estimate_samples * self.config.kl_loss_coef + (kl_estimate_base_weight + kl_estimate_base_bias) * self.config.kl_base_loss_coef

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class FlowMFScalingLinear(VariationalLayer):
    def __init__(self, config, in_features, out_features, initial_posterior_variance=1e-3, epsilon=1e-8, name="", writer=None):
        super().__init__(epsilon)
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.ipv = initial_posterior_variance
        self.name = name
        self.writer = writer
        self.step = 0

        # priors are not optimizable parameters - all means and log-variances are zero
        self.register_buffer('prior_W_means', torch.zeros(out_features, in_features))
        self.register_buffer('prior_W_log_vars', torch.zeros(out_features, in_features))
        self.register_buffer('prior_b_means', torch.zeros(out_features))
        self.register_buffer('prior_b_log_vars', torch.zeros(out_features))

        self.posterior_W_means = Parameter(torch.empty_like(self._buffers['prior_W_means'], requires_grad=True))
        self.posterior_b_means = Parameter(torch.empty_like(self._buffers['prior_b_means'], requires_grad=True))
        self.posterior_W_log_vars = Parameter(torch.empty_like(self._buffers['prior_W_log_vars'], requires_grad=True))
        self.posterior_b_log_vars = Parameter(torch.empty_like(self._buffers['prior_b_log_vars'], requires_grad=True))

        self._initialize_posteriors()

        self.flow_prior = StandardGaussianPrior()
        self.flow_posterior = InvertibleNetwork(config, out_features)
    
    def _initialize_posteriors(self):
        # posteriors on the other hand are optimizable parameters - means are normally distributed, log_vars
        # have some small initial value

        torch.nn.init.normal_(self.posterior_W_means, mean=0, std=0.1)
        torch.nn.init.uniform_(self.posterior_b_means, -0.1, 0.1)
        torch.nn.init.constant_(self.posterior_W_log_vars, math.log(self.ipv))
        torch.nn.init.constant_(self.posterior_b_log_vars, math.log(self.ipv))
    
    def _sample_gaussian_parameters(self):
        # obtained sampled weights and biases using local reparameterization trick
        w_epsilons = torch.randn_like(self.posterior_W_means)
        b_epsilons = torch.randn_like(self.posterior_b_means)

        w = self.posterior_W_means + torch.mul(w_epsilons, torch.exp(0.5 * self.posterior_W_log_vars))
        b = self.posterior_b_means + torch.mul(b_epsilons, torch.exp(0.5 * self.posterior_b_log_vars))

        return w, b
    
    def gaussian_kl_divergence(self) -> torch.Tensor:
        """ Returns KL(posterior, prior) for the parameters of this layer. """
        # obtain flattened means, log variances, and variances of the prior distribution
        prior_means = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_means'], (-1,)),
             torch.reshape(self._buffers['prior_b_means'], (-1,)))),
            requires_grad=False
        )
        prior_log_vars = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_log_vars'], (-1,)),
             torch.reshape(self._buffers['prior_b_log_vars'], (-1,)))),
            requires_grad=False
        )
        prior_vars = torch.exp(prior_log_vars)

        # obtain flattened means, log variances, and variances of the approximate posterior distribution
        posterior_means = torch.cat(
            (torch.reshape(self.posterior_W_means, (-1,)),
             torch.reshape(self.posterior_b_means, (-1,))),
        )
        posterior_log_vars = torch.cat(
            (torch.reshape(self.posterior_W_log_vars, (-1,)),
             torch.reshape(self.posterior_b_log_vars, (-1,))),
        )
        posterior_vars = torch.exp(posterior_log_vars)

        # compute kl divergence (this computation is valid for multivariate diagonal Gaussians)
        kl_elementwise = posterior_vars / (prior_vars + self.epsilon) + \
                         torch.pow(prior_means - posterior_means, 2) / (prior_vars + self.epsilon) - \
                         1 + prior_log_vars - posterior_log_vars

        return 0.5 * kl_elementwise.sum()
    
    def start_variational_phase(self):
        prior_norm = torch.norm(self.posterior_W_means.clone().detach(), p=2, dim=1)
        self.flow_prior.add_center(prior_norm)

    def forward(self, x, sample_parameters=True):
        """ Produces module output on an input. """
        
        if sample_parameters:
            w, b = self._sample_gaussian_parameters()
            out_scales = self.flow_posterior()[0]
            w = (w / torch.norm(w, p=2, dim=1)[:,None]) * out_scales[:,None]
            return F.linear(x, w, b)
        else:
            return F.linear(x, self.posterior_W_means, self.posterior_b_means)

    def reset_for_next_task(self):
        """ Overwrites the current prior with the current posterior. """
        self.flow_prior = copy.deepcopy(self.flow_posterior)
        self._buffers['prior_W_means'].data.copy_(self.posterior_W_means.data)
        self._buffers['prior_W_log_vars'].data.copy_(self.posterior_W_log_vars.data)
        self._buffers['prior_b_means'].data.copy_(self.posterior_b_means.data)
        self._buffers['prior_b_log_vars'].data.copy_(self.posterior_b_log_vars.data)
    
    def get_logs(self):
        return self.kl_divergence(return_log=True)

    def kl_divergence(self, return_log=False) -> torch.Tensor:
        """ Monte carlo estimate of the KL divergence for this layer """
        # obtain flattened means, log variances, and variances of the prior distribution
        kl_estimate = 0
        sample = self.flow_posterior(self.config.num_kl_estimate_samples)
        prior_log_probs = self.flow_prior.log_prob(sample)
        if len(prior_log_probs.shape) == 1:
            prior_log_probs = prior_log_probs.unsqueeze(1)
        kl_estimate -= prior_log_probs.sum(1).mean()
        kl_estimate += self.flow_posterior.log_prob(sample).mean()

        gaussian_kl_divergence = self.gaussian_kl_divergence()
        
        if return_log:
            return {
                "kl_flow": kl_estimate / self.config.num_kl_estimate_samples * self.config.kl_loss_coef,
                "kl_gaussian": gaussian_kl_divergence,
                # "abs_mean_weight": torch.abs(self.base_weights.weight).mean(),
                # "abs_mean_bias": torch.abs(self.base_weights.bias).mean()
            }
        
        # return gaussian_kl_divergence

        return kl_estimate / self.config.num_kl_estimate_samples * self.config.kl_loss_coef + gaussian_kl_divergence

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class CompleteFlowScalingLinear(VariationalLayer):

    def __init__(self, config, in_features, out_features, initial_posterior_variance=1e-3, epsilon=1e-8):
        super().__init__(epsilon)
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.ipv = initial_posterior_variance

        # priors are not optimizable parameters - all means and log-variances are zero
        self.mle_linear = nn.Linear(in_features, out_features, bias=True)
        self.prior = StandardGaussianPrior()
        self.posterior_net = InvertibleNetwork(config, (out_features + in_features)*self.config.complete_flow_rank)

    def posterior(self, num_samples=1, get_direct=False):
        r = self.config.complete_flow_rank
        direct_sample = self.posterior_net(num_samples)
        A = direct_sample[:, -self.in_features*r:].reshape(-1, self.in_features, r)
        B = direct_sample[:, :-self.in_features*r].reshape(-1, self.out_features, r)
        sample = einops.einsum(A, B, 'n i r, n o r -> n o i')
        if get_direct: return sample,direct_sample
        return sample

    def forward(self, x, sample_parameters=True):
        """ Produces module output on an input. """
        
        if sample_parameters:
            w = self.posterior()[0] + self.mle_linear.weight
            return F.linear(x, w, self.mle_linear.bias)
        else:
            return self.mle_linear(x)

    def reset_for_next_task(self):
        """ Overwrites the current prior with the current posterior. """
        self.prior = copy.deepcopy(self.posterior)

    def kl_divergence(self) -> torch.Tensor:
        """ Monte carlo estimate of the KL divergence for this layer """
        # obtain flattened means, log variances, and variances of the prior distribution
        kl_estimate = 0
        sample,direct_sample = self.posterior(self.config.num_kl_estimate_samples, get_direct=True)
        prior_log_probs = self.prior.log_prob(direct_sample)
        if len(prior_log_probs.shape) == 1:
            prior_log_probs = prior_log_probs.unsqueeze(1)
        kl_estimate -= prior_log_probs.sum(1).mean()
        kl_estimate += self.posterior_net.log_prob(direct_sample).mean()
        return kl_estimate / self.config.num_kl_estimate_samples


    def get_statistics(self) -> dict:
        sample_scales = self.posterior(5000)
        statistics = {
            'average_w_mean': sample_scales.mean(),
            'average_b_mean': torch.mean(self.base_weights.bias),
            'average_w_var': sample_scales.std(0).mean(),
            'average_b_var': torch.std(self.base_weights.bias)
        }

        return statistics

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
