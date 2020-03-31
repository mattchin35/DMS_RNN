import numpy as np
from torch import nn as nn
import torch
import os
import json
import config


class Abstract_Model(nn.Module):
    """Abstract Model class."""

    def __init__(self, save_path):
        super(Abstract_Model, self).__init__()

        if save_path is None:
            save_path = os.getcwd()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.model_config = None
        self.log_softmax = torch.nn.LogSoftmax()

    def save(self, name='model', epoch=None):
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print()

        model_path = os.path.join(save_path, name + '.pth')
        torch.save(self.state_dict(), model_path)
        self.save_config(save_path)

        print("Model saved in path: %s" % model_path)

    def load(self, name='model', epoch=None):
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
        save_path = os.path.join(save_path, name + '.pth')
        self.load_state_dict(torch.load(save_path))

        print("Model restored from path: {:s}".format(save_path))

    def save_config(self, save_path):
        model_config_dict = self.model_config.__dict__
        # input_config_dict = self.input_config.__dict__
        with open(os.path.join(save_path, 'model_config.json'), 'w') as f:
            json.dump(model_config_dict, f)
        # with open(os.path.join(save_path, 'input_config.json'), 'w') as f:
        #     json.dump(input_config_dict, f)

        with open(os.path.join(save_path, 'config.txt'), "w") as f:
            for k, v in model_config_dict.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')

            # for k, v in input_config_dict.items():
            #     f.write(str(k) + ' >>> ' + str(v) + '\n\n')


class Simple_Model(Abstract_Model):
    """Basic 1 layer RNN with no constraints."""

    def __init__(self, opts, isize, osize):
        super(Simple_Model, self).__init__(opts.save_path)

        self.hidden_size = opts.rnn_size
        self.batch_size = opts.batch_size
        self.config = opts

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
        i = self.i2h(input)
        h_effective = torch.mul(self.h_w, self.h_mask)
        h = torch.matmul(hidden, h_effective)
        noise = self.config.noise * torch.normal(mean=torch.zeros(self.batch_size, self.hidden_size))
        hidden = hidden * (1.- self.config.dt / self.config.tau) + \
                 torch.relu(i + h + self.h_b + noise) * self.config.dt / self.config.tau
        out = self.h2o(hidden)
        return hidden, out

    def initialZeroState(self):
        return torch.zeros(self.batch_size, self.hidden_size)


class EI_Model(Abstract_Model):
    def __init__(self, opts, isize, osize):
        super(EI_Model, self).__init__(opts.save_path)

        self.hidden_size = opts.rnn_size
        self.batch_size = opts.batch_size
        self.config = opts

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
        i = torch.matmul(input, torch.abs(self.i2h))

        _h_effective = torch.abs(torch.mul(self.h_w, self.h_mask))
        h_effective = torch.matmul(self.ei_mask, _h_effective)

        h = torch.matmul(hidden, h_effective)
        noise = self.config.noise * torch.normal(mean=torch.zeros(self.batch_size, self.hidden_size))
        hidden = torch.relu(i + h + self.h_b + noise)
        hidden = hidden * (1. - self.config.dt / self.config.tau) + \
                 torch.relu(i + h + self.h_b + noise) * self.config.dt / self.config.tau

        h2o_effective = torch.matmul(self.ei_mask, torch.abs(self.h2o_w))
        out = torch.matmul(hidden, h2o_effective) + self.h2o_b
        return hidden, out

    def initialZeroState(self):
        return torch.zeros(self.batch_size, self.hidden_size)


class Three_Layer_Model(Abstract_Model):
    """3 layer RNN with no constraints."""
    def __init__(self, opts, isize, osize):
        super(Three_Layer_Model, self).__init__(opts.save_path)

        self.hidden_size = opts.rnn_size
        self.batch_size = opts.batch_size
        self.opts = opts

        self.i_to_h0 = nn.Linear(isize, opts.rnn_size[0])
        self.h0_b = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size[0]))
        self.h0_w = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size[0], opts.rnn_size[0]))

        self.h0_to_h1 = nn.Linear(isize, opts.rnn_size[1])
        self.h1_b = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size[1]))
        self.h1_w = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size[1], opts.rnn_size[1]))

        self.h1_to_h2 = nn.Linear(isize, opts.rnn_size[2])
        self.h2_b = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size[2]))
        self.h2_w = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size[2], opts.rnn_size[2]))

        mask = (1 - np.eye(opts.rnn_size[0], opts.rnn_size[0])).astype(np.float32)
        h0_mask = torch.nn.Parameter(torch.from_numpy(mask), requires_grad=False)
        self.h0_mask = h0_mask

        mask = (1 - np.eye(opts.rnn_size[1], opts.rnn_size[1])).astype(np.float32)
        h1_mask = torch.nn.Parameter(torch.from_numpy(mask), requires_grad=False)
        self.h1_mask = h1_mask

        mask = (1 - np.eye(opts.rnn_size[2], opts.rnn_size[2])).astype(np.float32)
        h2_mask = torch.nn.Parameter(torch.from_numpy(mask), requires_grad=False)
        self.h2_mask = h2_mask

        self.h2_to_o = torch.nn.Linear(opts.rnn_size[2], osize)

    def forward(self, input, hidden):
        i0 = self.i_to_h0(input)
        h0 = torch.matmul(hidden[0], torch.mul(self.h0_w, self.h0_mask))
        hidden[0] = torch.relu(i0 + h0 + self.h1_b)

        i1 = self.h0_to_h1(input)
        h1 = torch.matmul(hidden[1], torch.mul(self.h1_w, self.h1_mask))
        hidden[1] = torch.relu(i1 + h1 + self.h2_b)

        i2 = self.h1_to_h2(input)
        h2 = torch.matmul(hidden[2], torch.mul(self.h2_w, self.h2_mask))
        hidden[2] = torch.relu(i2 + h2 + self.h2_b)

        out = self.h2_to_o(hidden[2])
        return hidden, out

    def initialZeroState(self):
        return torch.zeros(self.batch_size, self.hidden_size[0]), \
               torch.zeros(self.batch_size, self.hidden_size[1]), \
               torch.zeros(self.batch_size, self.hidden_size[2])


class Constrained_Model(Abstract_Model):
    """Restrict the input and output connections to certain neurons in a 1 layer RNN."""
    def __init__(self, opts, isize, osize):
        super(Constrained_Model, self).__init__(opts.save_path)
        assert opts.pir_size > 0
        assert opts.alm_size > 0
        assert opts.pir_size + opts.alm_size <= opts.rnn_size

        self.hidden_size = opts.rnn_size
        self.batch_size = opts.batch_size
        self.config = opts

        self.i2pir = nn.Linear(isize, opts.pir_size)
        self.alm2o = nn.Linear(opts.alm_size, osize)
        self.hidden_pad = torch.zeros((opts.batch_size, opts.rnn_size - opts.pir_size))

        self.h_w = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size, opts.rnn_size))
        self.h_b = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size))

        mask = np.ones((opts.rnn_size, opts.rnn_size)).astype(np.float32)
        np.fill_diagonal(mask, 0)
        mask[opts.state_size:, opts.state_size:] = 0
        mask = torch.from_numpy(mask)
        h_mask = torch.nn.Parameter(mask, requires_grad=False)
        self.h_mask = h_mask

    def forward(self, input, hidden):
        i_pir = self.i2pir(input)
        i = torch.cat((i_pir, self.hidden_pad), dim=1)

        h_effective = torch.mul(self.h_w, self.h_mask)
        h = torch.matmul(hidden, h_effective)
        hidden = torch.relu(i + h + self.h_b)

        out = self.alm2o(hidden[:, -self.config.alm_size:])
        return hidden, out

    def initialZeroState(self):
        return torch.zeros(self.batch_size, self.hidden_size)




def load_config(save_path, epoch=None):
    if epoch is not None:
        save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))

    with open(os.path.join(save_path, 'net.pth'), 'r') as f:
        config_dict = json.load(f)

    c = config.modelConfig()
    for key, val in config_dict.items():
        setattr(c, key, val)
    return c