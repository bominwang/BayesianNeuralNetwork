import typing
import torch
import torch.nn as nn
import numpy as np


class GaussianVariational(nn.Module):
    # Gaussian Variational family ---> checked
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor) -> None:
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)

        self.w = None
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log(1 + torch.exp(self.rho))

    def sample(self) -> torch.Tensor:
        device = self.mu.device
        epsilon = self.normal.sample(self.mu.size()).to(device)
        self.w = self.mu + self.sigma.to(device) * epsilon
        return self.w

    def log_posterior(self) -> torch.Tensor:
        # if self.w is None:
        #     return ValueError('self.w must have a value')
        log_const = np.log(np.sqrt(2 * np.pi))
        log_exp = ((self.w - self.mu) ** 2) / (2 * self.sigma ** 2)
        log_posterior = -log_const - torch.log(self.sigma) - log_exp
        return log_posterior.sum()


class ScaleMixture(nn.Module):
    # Scale Mixture Prior ---> checked
    def __init__(self, pi: float, sigma1: float, sigma2: float) -> None:
        super().__init__()

        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.normal1 = torch.distributions.Normal(0, sigma1)
        self.normal2 = torch.distributions.Normal(0, sigma2)

    def log_prior(self, w: torch.Tensor) -> torch.Tensor:
        likelihood_n1 = torch.exp(self.normal1.log_prob(w))
        likelihood_n2 = torch.exp(self.normal2.log_prob(w))

        prob = (self.pi * likelihood_n1 + (1 - self.pi) * likelihood_n2) + 1e-6
        log_prob = torch.log(prob).sum()
        return log_prob


def minibatch_weight(batch_idx: int, num_batches: int) -> float:
    # minibatch weight ---> checked
    return 2 ** (num_batches - batch_idx) / (2 ** num_batches - batch_idx)


class BayesianModule(nn.Module):
    # Base class for Bayesian Neural Network ---> checked
    def __init__(self):
        super().__init__()

    def kld(self, *args):
        raise NotImplementedError('BayesianModule::kld()')


class BayesianLinear(BayesianModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 prior_pi: typing.Optional[float] = 0.5,
                 prior_sigma1: typing.Optional[float] = 1.0,
                 prior_sigma2: typing.Optional[float] = 0.0025) -> None:
        super().__init__()

        # self.posterior_mu_init = 0
        # self.posterior_rho_init = -7.0

        # w_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(self.posterior_mu_init, 0.1))
        # w_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(self.posterior_rho_init, 0.1))
        #
        # bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(self.posterior_mu_init, 0.1))
        # bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(self.posterior_rho_init, 0.1))
        w_mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        w_rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)

        bias_mu = torch.empty(out_features).uniform_(-0.2, 0.2)
        bias_rho = torch.empty(out_features).uniform_(-5.0, -4.0)

        self.w_posterior = GaussianVariational(w_mu, w_rho)
        self.bias_posterior = GaussianVariational(bias_mu, bias_rho)

        self.w_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        self.bias_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)

        self.kl_divergence = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        w = self.w_posterior.sample()
        b = self.bias_posterior.sample()

        log_prior_w = self.w_prior.log_prior(w)
        log_prior_b = self.bias_prior.log_prior(b)

        log_posterior_w = self.w_posterior.log_posterior()
        log_posterior_b = self.bias_posterior.log_posterior()

        log_prior = log_prior_w + log_prior_b

        log_posterior = log_posterior_w + log_posterior_b

        self.kl_divergence = self.kld(log_prior=log_prior, log_posterior=log_posterior)

        return nn.functional.linear(x, w, b)

    def kld(self, log_prior: torch.Tensor, log_posterior: torch.Tensor) -> torch.Tensor:
        return log_posterior - log_prior


def variational_approximation(model: nn.Module) -> nn.Module:
    # Adds Variational inference functionality to nn.Module ---> checked
    def kl_divergence(self) -> torch.Tensor:

        kl = 0
        for module in self.modules():
            if isinstance(module, BayesianModule):
                kl += module.kl_divergence

        return kl

    setattr(model, 'kl_divergence', kl_divergence)

    def elbo(self,
             inputs: torch.Tensor,
             targets: torch.Tensor,
             criterion: typing.Any,
             n_samples: int,
             w_complexity:typing.Optional[float] = 1.0
             ) -> torch.Tensor:

        loss = 0
        for sample in range(n_samples):
            outputs = self(inputs)
            loss += criterion(outputs, targets)
            loss += self.kl_divergence() * w_complexity

        return loss / n_samples

    setattr(model, 'elbo', elbo)

    return model
