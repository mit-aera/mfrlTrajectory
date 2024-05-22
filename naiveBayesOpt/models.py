#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os
from matplotlib import pyplot as plt
from pyDOE import lhs

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F

import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.models import AbstractVariationalGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood, SoftmaxLikelihood
from gpytorch.models.deep_gps import AbstractDeepGPLayer, AbstractDeepGP, DeepLikelihood
from gpytorch.lazy import BlockDiagLazyTensor, lazify
from scipy.special import erf, expit

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *

class MFDeepGPLayer(AbstractDeepGPLayer):
    def __init__(self, input_dims, output_dims, prev_dims=0, num_inducing=512, inducing_points=None, prev_layer=None):
        self.prev_dims = prev_dims
        input_all_dims = input_dims + prev_dims
        
        # TODO
        if inducing_points is None:
            if output_dims is None:
                inducing_points = torch.randn(num_inducing, input_all_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_all_dims)
        
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(MFDeepGPLayer, self).__init__(variational_strategy, input_all_dims, output_dims)
        
        self.mean_module = ConstantMean(batch_size=output_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_size=output_dims, ard_num_dims=input_dims),
            batch_size=output_dims, ard_num_dims=None
        )
        
        self.prev_layer = prev_layer
        if prev_dims > 0:
            self.covar_module_corr = ScaleKernel(
                RBFKernel(batch_size=output_dims, ard_num_dims=input_dims),
                batch_size=output_dims, ard_num_dims=None
            )
            self.covar_module_prev = ScaleKernel(
                RBFKernel(batch_size=output_dims, ard_num_dims=None),
                batch_size=output_dims, ard_num_dims=None
            )
            self.covar_module_linear = ScaleKernel(
                LinearKernel(batch_size=output_dims, ard_num_dims=None)
            )
    
    def covar(self, x):
        x_input = torch.index_select(x, -1, torch.arange(self.prev_dims,self.input_dims).long().cuda())
        x_prev = torch.index_select(x, -1, torch.arange(self.prev_dims).long().cuda())
        covar_x = self.covar_module(x_input)
        if self.prev_dims > 0:
            k_corr = self.covar_module_corr(x_input)
            k_prev = self.covar_module_prev(x_prev)
#             k_prev = self.prev_layer.covar(x_input)
            k_linear = self.covar_module_linear(x_prev)
            covar_x += k_corr*(k_prev + k_linear)
#             covar_x = k_corr*(k_prev)
            
        return covar_x

    def forward(self, x):
        # https://github.com/amzn/emukit/blob/master/emukit/examples/multi_fidelity_dgp/multi_fidelity_deep_gp.py
        x_input = torch.index_select(x, -1, torch.arange(self.prev_dims,self.input_dims).long().cuda())
        mean_x = self.mean_module(x_input) # self.linear_layer(x).squeeze(-1)
        covar_x = self.covar(x)
            
        return MultivariateNormal(mean_x, covar_x)
    
    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()
#                 if 'eval' in kwargs:
#                     if kwargs['eval']:
#                         x = x.mean
#                     else:
#                         x = x.rsample()
#                 else:
#                     x = x.rsample()

                processed_inputs = [
                    inp.unsqueeze(0).expand(x.shape[0], *inp.shape)
                    for inp in other_inputs
                ]
            else:
                processed_inputs = [
                    inp for inp in other_inputs
                ]
            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

class MFDeepGPC(AbstractDeepGP):
    def __init__(self, train_x, train_y, num_inducing=512, input_uc=0):
        super().__init__()
        
        num_fidelity = len(train_x)
        train_x_shape = train_x[0].shape
        
        # Generate Inducing points - TODO check higher fidelity inducing points
        train_z = []
        
        i_z = torch.randperm(train_x[0].size(0)).cuda()[:num_inducing]
        z_low = train_x[0][i_z, :]
        setattr(self, 'train_z_' + str(0), z_low)
        train_z.append(z_low)
        for i in range(1,num_fidelity):
            i_z_low = torch.randperm(train_x[i-1].size(0)).cuda()[:num_inducing]
            z_high = torch.cat([train_x[i-1][i_z_low, :], train_y[i-1][i_z_low].unsqueeze(-1)], axis=1).unsqueeze(0)
            setattr(self, 'train_z_' + str(i), z_high)
            train_z.append(z_high)
        
        # Generate Multifidelity layers
        self.layers = []
        layer = MFDeepGPLayer(
            input_dims=train_x_shape[-1],
            output_dims=1,
            prev_dims=input_uc,
            num_inducing=num_inducing,
            inducing_points=train_z[0]
        )
        setattr(self, 'layer_' + str(0), layer)
        self.layers.append(layer)
        
        for i in range(1,num_fidelity):
            layer = MFDeepGPLayer(
                input_dims=train_x_shape[-1],
                output_dims=1,
                prev_dims=1,
                num_inducing=num_inducing,
                inducing_points=train_z[i],
                prev_layer=self.layers[i-1]
            )
            setattr(self, 'layer_' + str(i), layer)
            self.layers.append(layer)
        
        self.likelihood = DeepLikelihood(BernoulliLikelihood())
    
    def forward(self, inputs, fidelity=2, eval=False):
        val = self.layers[0](inputs, eval=eval)
        for layer in self.layers[1:fidelity]:
            val = layer(val, inputs, eval=eval)
        val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
        return val
    
    def predict(self, x, fidelity=2):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)
        
        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])
        
        # return preds
        return self.likelihood.base_likelihood(val).mean.ge(0.5).cpu().numpy()
    
    def predict_proba(self, x, fidelity=2, return_std=False):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)

        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])

        pred_means = val.mean.cpu().numpy()
        pred_vars = val.variance.cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means
        
        bern = self.likelihood.base_likelihood(val)
        pi_star = bern.probs.cpu().numpy()
        
        if return_std:
            return f_star_min, np.sqrt(var_f_star)
        else:
            return f_star_min, np.sqrt(var_f_star), np.vstack((1 - pi_star, pi_star)).T
        
    def predict_proba_MF(self, x, fidelity=1, C_L=1., C_H=10., beta=0.05, return_std=False, return_all=False):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)
        
        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])

        pred_means = val.mean.detach().cpu().numpy()
        pred_vars = val.variance.detach().cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means - beta*np.sqrt(pred_vars)
        f_star_min_i = pred_means
        
        val_mf = MultivariateNormal(val.mean-beta*torch.sqrt(val.variance), val.lazy_covariance_matrix)
        bern = self.likelihood.base_likelihood(val_mf)
        pi_star = bern.probs.detach().cpu().numpy()
        
        bern_i = self.likelihood.base_likelihood(val)
        pi_star_i = bern_i.probs.detach().cpu().numpy()
        
        if return_all:
            if return_std:
                return f_star_min, np.sqrt(var_f_star), f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
            else:
                return np.vstack((1 - pi_star, pi_star)).T, f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
        else:
            if return_std:
                return f_star_min, np.sqrt(var_f_star)
            else:
                return np.vstack((1 - pi_star, pi_star)).T


class MFS2SDeepGPC(AbstractDeepGP):
    def __init__(self, inducing_points, feature_network, input_uc=0):
        super().__init__()
        self.feature_network = feature_network
        self.latent_size = inducing_points[0].shape[1]
        
        self.num_fidelity = len(inducing_points)
        
        # Generate Inducing points - TODO check higher fidelity inducing points        
        setattr(self, 'train_z_' + str(0), inducing_points[0])
        for i in range(1,self.num_fidelity):
            setattr(self, 'train_z_' + str(i), inducing_points[i])
        
        # Generate Multifidelity layers
        self.layers = []
        layer = MFDeepGPLayer(
            input_dims=self.latent_size,
            output_dims=1,
            prev_dims=input_uc,
            num_inducing=inducing_points[0].shape[0],
            inducing_points=inducing_points[0]
        )
        setattr(self, 'layer_' + str(0), layer)
        self.layers.append(layer)
        
        for i in range(1,self.num_fidelity):
            layer = MFDeepGPLayer(
                input_dims=self.latent_size,
                output_dims=1,
                prev_dims=1,
                num_inducing=inducing_points[i].shape[0],
                inducing_points=inducing_points[i].unsqueeze(0),
                prev_layer=self.layers[i-1]
            )
            setattr(self, 'layer_' + str(i), layer)
            self.layers.append(layer)
        
        self.likelihood = DeepLikelihood(BernoulliLikelihood())
        
    def get_inducing_points(self):
        inducing_points = []
        for i in range(self.num_fidelity):
            inducing_points.append(self.layers[i].variational_strategy.inducing_points.detach().cpu().squeeze().numpy())
        return inducing_points
    
    def forward(self, inputs, fidelity=2, eval=False):
        inputs_t, _ = self.feature_network(torch.swapaxes(inputs[0], 0, 1), inputs[1]) # data, sorted length
        val = self.layers[0](inputs_t, eval=eval)
        for layer in self.layers[1:fidelity]:
            val = layer(val, inputs_t, eval=eval)
        val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
        return val
    
    def forward_aux(self, inputs, fidelity=2, eval=False):
        if eval:
            inputs_t, _ = self.feature_network(torch.swapaxes(inputs[0], 0, 1), inputs[1]) # data, sorted length
            val = self.layers[0](inputs_t, eval=eval)
            for layer in self.layers[1:fidelity]:
                val = layer(val, inputs_t, eval=eval)
            val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
            return val
        else:
            o_wp, o_time, o_snapw, inputs_t, logv = self.feature_network.forward_full(torch.swapaxes(inputs[0], 0, 1), inputs[1]) # data, sorted length
            val = self.layers[0](inputs_t, eval=eval)
            for layer in self.layers[1:fidelity]:
                val = layer(val, inputs_t, eval=eval)
            val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
            return val, o_wp, o_time, o_snapw, inputs_t, logv
    
    def predict(self, x, fidelity=2):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)
        
        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])
        
        # return preds
        return self.likelihood.base_likelihood(val).mean.ge(0.5).cpu().numpy()
    
    def predict_proba(self, x, fidelity=2, return_std=False):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)

        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])

        pred_means = val.mean.detach().cpu().numpy()
        pred_vars = val.variance.detach().cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means
        
        bern = self.likelihood.base_likelihood(val)
        pi_star = bern.probs.detach().cpu().numpy()
        
        if return_std:
            return f_star_min, np.sqrt(var_f_star)
        else:
            return f_star_min, np.sqrt(var_f_star), np.vstack((1 - pi_star, pi_star)).T
        
    def predict_proba_MF(self, x, fidelity=1, beta=0.05, return_std=False, return_all=False):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)
        
        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])

        pred_means = val.mean.detach().cpu().numpy()
        pred_vars = val.variance.detach().cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means - beta*np.sqrt(pred_vars)
        f_star_min_i = pred_means
        
        val_mf = MultivariateNormal(val.mean-beta*torch.sqrt(val.variance), val.lazy_covariance_matrix)
        bern = self.likelihood.base_likelihood(val_mf)
        pi_star = bern.probs.detach().cpu().numpy()
        
        bern_i = self.likelihood.base_likelihood(val)
        pi_star_i = bern_i.probs.detach().cpu().numpy()
        
        if return_all:
            if return_std:
                return f_star_min, np.sqrt(var_f_star), f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
            else:
                return np.vstack((1 - pi_star, pi_star)).T, f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
        else:
            if return_std:
                return f_star_min, np.sqrt(var_f_star)
            else:
                return np.vstack((1 - pi_star, pi_star)).T

class MFCondS2SDeepGPC(AbstractDeepGP):
    def __init__(self, inducing_points, feature_network, input_uc=0):
        super().__init__()
        self.feature_network = feature_network
        self.latent_size = inducing_points[0].shape[1]
        
        self.num_fidelity = len(inducing_points)
        
        # Generate Inducing points - TODO check higher fidelity inducing points        
        setattr(self, 'train_z_' + str(0), inducing_points[0])
        for i in range(1,self.num_fidelity):
            # z_high = torch.cat([inducing_points_z[i], inducing_poings_y[i-1].unsqueeze(-1)], axis=1).unsqueeze(0)
            setattr(self, 'train_z_' + str(i), inducing_points[i])
        
        # Generate Multifidelity layers
        self.layers = []
        layer = MFDeepGPLayer(
            input_dims=self.latent_size,
            output_dims=1,
            prev_dims=input_uc,
            num_inducing=inducing_points[0].shape[0],
            inducing_points=inducing_points[0]
        )
        setattr(self, 'layer_' + str(0), layer)
        self.layers.append(layer)
        
        for i in range(1,self.num_fidelity):
            layer = MFDeepGPLayer(
                input_dims=self.latent_size,
                output_dims=1,
                prev_dims=1,
                num_inducing=inducing_points[i].shape[0],
                inducing_points=inducing_points[i].unsqueeze(0),
                prev_layer=self.layers[i-1]
            )
            setattr(self, 'layer_' + str(i), layer)
            self.layers.append(layer)
        
        self.likelihood = DeepLikelihood(BernoulliLikelihood())
    
    def get_inducing_points(self):
        inducing_points = []
        for i in range(self.num_fidelity):
            inducing_points.append(self.layers[i].variational_strategy.inducing_points.detach().cpu().squeeze().numpy())
        return inducing_points
    
    def forward(self, inputs, fidelity=2, eval=False):
        inputs_t, _ = self.feature_network(torch.swapaxes(inputs[0], 0, 1), inputs[1], initial=inputs[2]) # data, sorted length, bpoly
        # print("Input shape")
        # print(inputs_t.shape)
        # print(inputs_t)
        
        val = self.layers[0](inputs_t, eval=eval)
        for layer in self.layers[1:fidelity]:
            val = layer(val, inputs_t, eval=eval)
        val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
        return val

    def forward_train(self, inputs, fidelity=2, eval=False):
        inputs_t, _ = self.feature_network(torch.swapaxes(inputs[0], 0, 1), inputs[1], initial=inputs[2]) # data, sorted length, bpoly
        # print("Input shape")
        # print(inputs_t.shape)
        # print(inputs_t)
        
        outputs = []
        val = self.layers[0](inputs_t, eval=eval)
        outputs.append(MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix))
        for layer in self.layers[1:fidelity]:
            val = layer(val, inputs_t, eval=eval)
            outputs.append(MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix))
        # val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
        return outputs
    
    def forward_grad_penalty(self, inputs, fidelity=2, eval=False):
        x_batch = torch.autograd.Variable(torch.swapaxes(inputs[0], 0, 1), requires_grad=True)
        len_batch = torch.autograd.Variable(inputs[1], requires_grad=True)
        bpoly_batch = torch.autograd.Variable(inputs[2], requires_grad=True)
        inputs_t, _ = self.feature_network(x_batch, len_batch, initial=bpoly_batch)
        gradients = torch.autograd.grad(
            outputs=inputs_t, inputs=[x_batch, bpoly_batch],
            grad_outputs=torch.ones(inputs_t.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        val = self.layers[0](inputs_t, eval=eval)
        for layer in self.layers[1:fidelity]:
            val = layer(val, inputs_t, eval=eval)
        val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
        return val, gradient_penalty
    
    def forward_aux(self, inputs, fidelity=2, eval=False):
        if eval:
            inputs_t, _ = self.feature_network(torch.swapaxes(inputs[0], 0, 1), inputs[1], initial=inputs[2]) # data, sorted length, bpoly
            val = self.layers[0](inputs_t, eval=eval)
            for layer in self.layers[1:fidelity]:
                val = layer(val, inputs_t, eval=eval)
            val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
            return val
        else:
            o_wp, o_time, o_snapw, mean, logv, inputs_t = self.feature_network.forward_full(torch.swapaxes(inputs[0], 0, 1), inputs[1], initial=inputs[2]) # data, sorted length, bpoly
            val = self.layers[0](inputs_t, eval=eval)
            for layer in self.layers[1:fidelity]:
                val = layer(val, inputs_t, eval=eval)
            val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
            return val, o_wp, o_time, o_snapw, mean, logv
    
    def predict(self, x, fidelity=2):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)
        
        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])
        
        # return preds
        return self.likelihood.base_likelihood(val).mean.ge(0.5).cpu().numpy()
    
    def predict_proba(self, x, fidelity=2, return_std=False):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)

        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])

        pred_means = val.mean.detach().cpu().numpy()
        pred_vars = val.variance.detach().cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means
        
        bern = self.likelihood.base_likelihood(val)
        pi_star = bern.probs.detach().cpu().numpy()
        
        if return_std:
            return f_star_min, np.sqrt(var_f_star)
        else:
            return f_star_min, np.sqrt(var_f_star), np.vstack((1 - pi_star, pi_star)).T
    
    def predict_proba_full(self, x, fidelity=2):
        inputs_t, _ = self.feature_network(torch.swapaxes(x[0], 0, 1), x[1], initial=x[2]) # data, sorted length, bpoly
        preds = self.layers[0](inputs_t, eval=eval)
        for layer in self.layers[1:fidelity]:
            preds = layer(preds, inputs_t, eval=eval)
        preds = MultivariateNormal(preds.mean.squeeze(-1), preds.lazy_covariance_matrix)

        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])

        pred_means = val.mean.detach().cpu().numpy()
        pred_vars = val.variance.detach().cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means
        
        bern = self.likelihood.base_likelihood(val)
        pi_star = bern.probs.detach().cpu().numpy()
        
        return f_star_min, np.sqrt(var_f_star), np.vstack((1 - pi_star, pi_star)).T, inputs_t

###########################################################################################
## ActiveMFDGP_manifold
###########################################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
        torch.nn.init.constant_(m.bias, 0)

class FeatureNetwork(nn.Module):
    def __init__(self, x_dim=2, h_dim=64, f_dim=16):
        super(FeatureNetwork, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, f_dim))
#         self.enc = nn.Sequential(
#             nn.Linear(x_dim, f_dim))
        
        self.fc_class = nn.Sequential(
            nn.Linear(f_dim, 1),
            nn.Sigmoid())
        self.apply(weights_init)
        
    def forward(self, x):
        z = self.enc(x)
        y_l = self.fc_class(z)
        return None, None, \
            (z, None), \
            None, y_l
    
class VAE(nn.Module):
    def __init__(self, x_dim=2, h_dim=64, f_dim=16):
        super(VAE, self).__init__()
        self.beta = 10000.0
        
        self.x_dim = x_dim
        self.enc = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
#         self.enc = nn.Sequential(
#             nn.Linear(x_dim, h_dim),
#             nn.ReLU())
        self.enc_mean = nn.Sequential(
            nn.Linear(h_dim, f_dim))
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, f_dim))
        
        self.dec = nn.Sequential(
            nn.Linear(f_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
#         self.dec = nn.Sequential(
#             nn.Linear(f_dim, h_dim),
#             nn.ReLU())
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())
        
        self.fc_class = nn.Sequential(
            nn.Linear(f_dim, 1),
            nn.Sigmoid())
        
        self.apply(weights_init)

    def encode(self, x):
        h1 = self.enc(x)
        return self.enc_mean(h1), self.enc_std(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.dec(z)
        return self.dec_mean(h3)

    def forward(self, x):
        kld_loss = 0
        nll_loss = 0

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_l = self.fc_class(z)
        recon_x = self.decode(z)
        
        kld_loss += self._kld_gauss(mu, logvar)
        nll_loss += self._nll_bernoulli(recon_x, x)*self.beta
        
        x_var = logvar.exp()
#         print("shape: {},{}".format(mu.shape,x_var.shape))
#         prRed(torch.max(mu))
#         prRed(torch.min(mu))
#         prRed(torch.max(torch.sqrt(x_var)))
#         prRed(torch.min(torch.sqrt(x_var)))
        return kld_loss, nll_loss, \
            (mu, x_var), \
            (z, recon_x), y_l

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _kld_gauss(self, mu, logvar):
        return 0.5 * torch.sum(mu.pow(2)+logvar.exp()-1-logvar)

    def _nll_bernoulli(self, recon_x, x):
        return F.mse_loss(recon_x, x, reduction='sum')

class cnnVAE(VAE):
    def __init__(self, x_dim=2, h_dim=64, f_dim=16):
        super(cnnVAE, self).__init__()
        
        self.x_dim = x_dim
        self.enc = nn.Sequential(
            nn.Conv1d(1, h_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.conv_dim = np.int(np.ceil(np.ceil(np.ceil(x_dim/2-2)/2-2)/2)-2)
        self.enc_mean = nn.Linear(self.conv_dim*h_dim, f_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.conv_dim*h_dim, f_dim),
            nn.Softplus())
        self.h_dim = h_dim
        
        self.dec_fc = nn.Linear(f_dim, self.conv_dim*h_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim, 1, kernel_size=4, stride=1, padding=2)
        )
        self.dec_mean = nn.Sigmoid()
        
        self.fc_class = nn.Sequential(
            nn.Linear(f_dim, 1),
            nn.Sigmoid())
        
        self.apply(weights_init)

    def encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze_(1)
        h1 = self.enc(x)
        return self.enc_mean(h1), self.enc_std(h1)
    
    def decode(self, z):
        h3_t = self.dec_fc(z)
        h3 = self.dec(h3_t.view(h3_t.size(0), self.h_dim, -1))
        return self.dec_mean(h3)

###########################################################################################
## ActiveMFDGP_imbalanced
###########################################################################################

class NN_UQ(nn.Module):
    def __init__(self, x_dim=2, h_dim=64, f_dim=16, droprates=[0.5,0.5,0.0]):
        super(NN_UQ, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.Dropout(droprates[0]),
            nn.ReLU(),
            nn.Dropout(droprates[1]),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(droprates[2]),
            nn.Linear(h_dim, f_dim))
        
        self.fc_class = nn.Sequential(
            nn.ReLU(),
            nn.Linear(f_dim, 1))
        
        self.fc_class_out = nn.Sigmoid()
        self.apply(weights_init)
        
    def forward(self, x):
        z = self.enc(x)
        y_l = self.fc_class(z)
        y_out = torch.sigmoid(y_l)
        return z, y_l, y_out

class NN_UQ_Feature:
    def __init__(self, x_dim=2, h_dim=64, f_dim=16, droprates=[0.5,0.5,0.0]):
        self.model = NN_UQ(x_dim, h_dim, f_dim, droprates).cuda()
        self.criterion = nn.BCELoss().cuda()
        self.x_dim = x_dim
        self.f_dim = f_dim
        
    def forward(self, inputs, labels):
        feature, outputs_raw, outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return loss
    
    def predict_mean(self, x):
        model = self.model.eval()
        feature, outputs_raw, output = model(x)
        model = self.model.train()
        return feature, outputs_raw, output
    
    def predict(self, x, T=1000):
        m, _, label = self.predict_mean(x)
        y_sum = torch.zeros(x.shape[0],self.f_dim).cuda()
        y_sig = torch.zeros(x.shape[0],self.f_dim,self.f_dim).cuda()
        for _ in range(T):
            f_, y_ = self.model(x)
            y_sum += f_
            y_sig += torch.einsum("ijk,ikl->ijl",(f_.unsqueeze(-1), f_.unsqueeze(-2)))
        v = y_sig/T - torch.einsum("ijk,ikl->ijl",(y_sum.unsqueeze(-1), y_sum.unsqueeze(-2)))/T/T
        return label, m, v
    
    def __call__(self, inputs):
        return predict(inputs)
    
    def parameters(self):
        return self.model.parameters()

class NN_UQ_Classifier:
    def __init__(self, x_dim=2, h_dim=64, f_dim=64, droprates=[0.5,0.5,0.5]):
        self.model = NN_UQ(x_dim, h_dim, f_dim, droprates).cuda()
        self.criterion = nn.BCELoss(reduction='sum').cuda()
        self.x_dim = x_dim
        self.f_dim = f_dim
        self.tau_inv = 0.02
        
    def forward(self, inputs, labels, return_output=False):
        feature, outputs_raw, outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        if return_output:
            return loss, outputs_raw, outputs
        else:
            return loss
    
    def predict_mean(self, x):
        model = self.model.eval()
        feature, outputs_raw, output = model(x)
        model = self.model.train()
        return feature, outputs_raw, output
    
    def predict(self, x, T=2000):
        m_f, m, label = self.predict_mean(x)
        y_sum = torch.zeros(x.shape[0],1).cuda()
        y_sig = torch.zeros(x.shape[0],1,1).cuda()
        for _ in range(T):
            f_, y_raw_, y_ = self.model(x)
            y_sum += y_raw_
            y_sig += torch.einsum("ijk,ikl->ijl",(y_raw_.unsqueeze(-1), y_raw_.unsqueeze(-2)))
        v = self.tau_inv + y_sig/T - torch.einsum("ijk,ikl->ijl",(y_sum.unsqueeze(-1), y_sum.unsqueeze(-2)))/T/T
        m.squeeze_()
        v.squeeze_()
        return label, m, v
    
    def predict_proba_MF(self, x, beta=1.0, return_std=False, return_all=True, T=2000):
        label, m, v = self.predict(x, T)
        
        var_f_star = v.data.cpu().numpy()
        f_star_min = m - beta*torch.sqrt(v)
        f_star_min_i = m
        
        pi_star = torch.sigmoid(f_star_min).data.cpu().numpy()
        pi_star_i = torch.sigmoid(f_star_min_i).data.cpu().numpy()
        
        f_star_min = f_star_min.data.cpu().numpy()
        f_star_min_i = f_star_min_i.data.cpu().numpy()
        
        if return_all:
            if return_std:
                return f_star_min, np.sqrt(var_f_star), f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
            else:
                return np.vstack((1 - pi_star, pi_star)).T, f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
        else:
            if return_std:
                return f_star_min, np.sqrt(var_f_star)
            else:
                return np.vstack((1 - pi_star, pi_star)).T
        
    def __call__(self, inputs):
        return predict(inputs)
    
    def parameters(self):
        return self.model.parameters()


class MFDeepGPCSingle(AbstractDeepGP):
    def __init__(self, train_x, train_y, num_inducing=512, input_uc=0):
        super().__init__()
        
        if input_uc == 0:
            num_fidelity = len(train_x)
        else:
            num_fidelity = len(train_x) - 1
        train_x_shape = train_x[0].shape
        
        # Generate Inducing points - TODO check higher fidelity inducing points
        train_z = []
        
        if input_uc == 0:
            i_z = torch.randperm(train_x[0].size(0)).cuda()[:num_inducing]
            z_low = train_x[0][i_z, :]
            setattr(self, 'train_z_' + str(0), z_low)
            train_z.append(z_low)
            
            for i in range(1,num_fidelity):
                i_z_low = torch.randperm(train_x[i-1].size(0)).cuda()[:num_inducing]
                z_high = torch.cat([train_x[i-1][i_z_low, :], train_y[i-1][i_z_low].unsqueeze(-1)], axis=1).unsqueeze(0)
                setattr(self, 'train_z_' + str(i), z_high)
                train_z.append(z_high)
        else:
            for i in range(num_fidelity):
                i_z_low = torch.randperm(train_x[i].size(0)).cuda()[:num_inducing]
                z_high = torch.cat([train_x[i][i_z_low, :], train_y[i][i_z_low].unsqueeze(-1)], axis=1).unsqueeze(0)
                setattr(self, 'train_z_' + str(i), z_high)
                train_z.append(z_high)
        
        # Generate Multifidelity layers
        self.layers = []
        layer = MFDeepGPLayer(
            input_dims=train_x_shape[-1],
            output_dims=1,
            prev_dims=input_uc,
            num_inducing=num_inducing,
            inducing_points=train_z[0]
        )
        setattr(self, 'layer_' + str(0), layer)
        self.layers.append(layer)
        
        for i in range(1,num_fidelity):
            layer = MFDeepGPLayer(
                input_dims=train_x_shape[-1],
                output_dims=1,
                prev_dims=1,
                num_inducing=num_inducing,
                inducing_points=train_z[i],
                prev_layer=self.layers[i-1]
            )
            setattr(self, 'layer_' + str(i), layer)
            self.layers.append(layer)
        
        self.likelihood = DeepLikelihood(BernoulliLikelihood())
    
    def forward(self, inputs, x_prev, fidelity=2, eval=False):
        val = self.layers[0](x_prev, inputs, eval=eval)
        for layer in self.layers[1:fidelity]:
            val = layer(val, inputs, eval=eval)
        val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
        return val
    
    def predict(self, x_prev, inputs, fidelity=2):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(inputs, x_prev, fidelity=fidelity, eval=True)
        
        val = MultivariateNormal(preds.mean, preds.lazy_covariance_matrix)
#         for i in range(1,preds.mean.shape[0]):
#             val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
#         val /= (preds.mean.shape[0])
        
        return self.likelihood.base_likelihood(val).mean.ge(0.5).cpu().numpy()
    
    def predict_proba(self, x_prev, inputs, fidelity=2, return_std=False):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(inputs, x_prev, fidelity=fidelity, eval=True)

        val = MultivariateNormal(preds.mean, preds.lazy_covariance_matrix)
#         for i in range(1,preds.mean.shape[0]):
#             val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
#         val /= (preds.mean.shape[0])

        pred_means = val.mean.cpu().numpy()
        pred_vars = val.variance.cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means
        
        bern = self.likelihood.base_likelihood(val)
        pi_star = bern.probs.cpu().numpy()
        
        if return_std:
            return f_star_min, np.sqrt(var_f_star)
        else:
            return f_star_min, np.sqrt(var_f_star), np.vstack((1 - pi_star, pi_star)).T
        
    def predict_proba_MF(self, x_prev, inputs, fidelity=1, C_L=1., C_H=10., beta=0.05, return_std=False, return_all=False):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(inputs, x_prev, fidelity=fidelity, eval=True)
        
        val = MultivariateNormal(preds.mean, preds.lazy_covariance_matrix)
#         for i in range(1,preds.mean.shape[0]):
#             val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
#         val /= (preds.mean.shape[0])
        
        pred_means = val.mean.cpu().numpy()
        pred_vars = val.variance.cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means - beta*np.sqrt(pred_vars)
        f_star_min_i = pred_means
        
        val_mf = MultivariateNormal(val.mean-beta*torch.sqrt(val.variance), val.lazy_covariance_matrix)
        bern = self.likelihood.base_likelihood(val_mf)
        pi_star = bern.probs.cpu().numpy()
        
        bern_i = self.likelihood.base_likelihood(val)
        pi_star_i = bern_i.probs.cpu().numpy()
        
        if return_all:
            if return_std:
                return f_star_min, np.sqrt(var_f_star), f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
            else:
                return np.vstack((1 - pi_star, pi_star)).T, f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
        else:
            if return_std:
                return f_star_min, np.sqrt(var_f_star)
            else:
                return np.vstack((1 - pi_star, pi_star)).T
