import numpy as np
from torch import nn as nn
import torch
import os
import json
import config
from torch_model import Abstract_Model


class XJW_Simple(Abstract_Model):
    """Paper-ready 1 layer RNN with no constraints. Separates hidden state and firing rate."""

    def __init__(self, opts, isize, osize):
        super(XJW_Simple, self).__init__(opts.save_path)

        self.hidden_size = opts.rnn_size
        self.batch_size = opts.batch_size
        self.config = opts
        self.time_const = self.config.dt / self.config.tau

        self.i2h = nn.Linear(isize, opts.rnn_size)
        self.h_b = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size))
        self.h_w = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size, opts.rnn_size))
        mask = np.ones((opts.rnn_size, opts.rnn_size)).astype(np.float32)
        np.fill_diagonal(mask, 0)
        mask = torch.from_numpy(mask)
        h_mask = torch.nn.Parameter(mask, requires_grad=False)
        self.h_mask = h_mask
        self.h2o = torch.nn.Linear(opts.rnn_size, osize)

    def forward(self, input, hidden):
        hprev, rate_prev = hidden
        i = self.i2h(input)
        h_effective = torch.mul(self.h_w, self.h_mask)
        hrec = torch.matmul(rate_prev, h_effective)
        # noise = np.sqrt(2 * self.time_const * self.config.network_noise ** 2) * \
        #         torch.normal(mean=torch.zeros(self.batch_size, self.hidden_size))
        noise = self.config.network_noise * torch.normal(mean=torch.zeros(self.batch_size, self.hidden_size))
        ht = hprev * (1.- self.time_const) + \
            (i + hrec + self.h_b) * self.time_const + noise
        rt = torch.relu(ht)
        out = self.h2o(rt)
        return (ht, rt), out

    def initialZeroState(self):
        return torch.zeros(self.batch_size, self.hidden_size)


class XJW_EI(Abstract_Model):
    def __init__(self, opts, isize, osize):
        super(XJW_EI, self).__init__(opts.save_path)

        self.hidden_size = opts.rnn_size
        self.batch_size = opts.batch_size
        self.config = opts
        self.time_const = self.config.dt / self.config.tau

        self.i2h = torch.nn.Parameter(.01 * torch.rand(isize, opts.rnn_size))

        target = 2
        alpha = 2
        nE = int(opts.rnn_size * opts.percent_E)
        nI = opts.rnn_size - nE
        E = np.random.gamma(shape=alpha, scale=target / (nE * alpha), size=[nE, opts.rnn_size])
        I = np.random.gamma(shape=alpha, scale=target / (nI * alpha), size=[nI, opts.rnn_size])
        EI = np.concatenate([E, I], axis=0).astype(np.float32)
        self.h_w = torch.nn.Parameter(torch.from_numpy(EI))
        self.h_b = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size))

        ei_mask = np.eye(opts.rnn_size).astype(np.float32)
        ei_mask[nE:] *= -1
        self.ei_mask = torch.nn.Parameter(torch.from_numpy(ei_mask), requires_grad=False)

        mask = np.ones((opts.rnn_size, opts.rnn_size)).astype(np.float32)
        np.fill_diagonal(mask, 0)
        mask = torch.from_numpy(mask)
        h_mask = torch.nn.Parameter(mask, requires_grad=False)
        self.h_mask = h_mask

        self.h2o_w = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size, osize))
        self.h2o_b = torch.nn.Parameter(.01 * torch.rand(osize))

    def forward(self, input, hidden):
        hprev, rate_prev = hidden
        i = torch.matmul(input, torch.abs(self.i2h))

        _h_effective = torch.abs(torch.mul(self.h_w, self.h_mask))
        h_effective = torch.matmul(self.ei_mask, _h_effective)

        hrec = torch.matmul(rate_prev, h_effective)
        noise = self.config.network_noise * torch.normal(mean=torch.zeros(self.batch_size, self.hidden_size))
        ht = hprev * (1. - self.config.dt / self.config.tau) + \
                 (i + hrec + self.h_b + noise) * self.config.dt / self.config.tau
        rt = torch.relu(ht)

        h2o_effective = torch.matmul(self.ei_mask, torch.abs(self.h2o_w))
        out = torch.matmul(rt, h2o_effective) + self.h2o_b
        return (ht, rt), out

    def initialZeroState(self):
        return torch.zeros(self.batch_size, self.hidden_size)

    def lesion(self, input, hidden, ix):
        hprev, rate_prev = hidden
        i = torch.matmul(input, torch.abs(self.i2h))

        _h_effective = torch.abs(torch.mul(self.h_w, self.h_mask))
        h_effective = torch.matmul(self.ei_mask, _h_effective)
        h_effective[ix.tolist(),:] *= 0

        hrec = torch.matmul(rate_prev, h_effective)
        noise = self.config.network_noise * torch.normal(mean=torch.zeros(self.batch_size, self.hidden_size))
        ht = hprev * (1. - self.config.dt / self.config.tau) + \
                 (i + hrec + self.h_b + noise) * self.config.dt / self.config.tau
        rt = torch.relu(ht)

        h2o_effective = torch.matmul(self.ei_mask, torch.abs(self.h2o_w))
        out = torch.matmul(rt, h2o_effective) + self.h2o_b
        return (ht, rt), out

    def stimulate(self, input, hidden, ix):
        hprev, rate_prev = hidden
        i = torch.matmul(input, torch.abs(self.i2h))

        _h_effective = torch.abs(torch.mul(self.h_w, self.h_mask))
        h_effective = torch.matmul(self.ei_mask, _h_effective)
        rate_prev[:, ix.tolist()] = .5

        hrec = torch.matmul(rate_prev, h_effective)
        noise = self.config.network_noise * \
                torch.normal(mean=torch.zeros(self.batch_size, self.hidden_size))
        ht = hprev * (1. - self.config.dt / self.config.tau) + \
                 (i + hrec + self.h_b + noise) * self.config.dt / self.config.tau
        rt = torch.relu(ht)

        h2o_effective = torch.matmul(self.ei_mask, torch.abs(self.h2o_w))
        out = torch.matmul(rt, h2o_effective) + self.h2o_b
        return (ht, rt), out
