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
        self.input_config = None

    def save(self, name='model', epoch=None):
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
        input_config_dict = self.input_config.__dict__
        with open(os.path.join(save_path, 'model_config.json'), 'w') as f:
            json.dump(model_config_dict, f)
        with open(os.path.join(save_path, 'input_config.json'), 'w') as f:
            json.dump(input_config_dict, f)

        with open(os.path.join(save_path, 'config.txt'), "w") as f:
            for k, v in model_config_dict.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')

            for k, v in input_config_dict.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')

class Simple_Model(Abstract_Model):
    def __init__(self, modelConfig, inputConfig, isize, osize):
        super(Simple_Model, self).__init__(modelConfig.save_path)

        self.hidden_size = modelConfig.rnn_size
        self.batch_size = modelConfig.batch_size
        self.model_config = modelConfig
        self.input_config = inputConfig

        self.i2h = nn.Linear(isize, modelConfig.rnn_size)
        self.h_b = torch.nn.Parameter(.01 * torch.rand(modelConfig.rnn_size))
        self.h_w = torch.nn.Parameter(.01 * torch.rand(modelConfig.rnn_size, modelConfig.rnn_size))
        mask = np.ones((modelConfig.rnn_size, modelConfig.rnn_size)).astype(np.float32)
        np.fill_diagonal(mask, 0)
        mask = torch.from_numpy(mask)
        h_mask = torch.nn.Parameter(mask, requires_grad=False)
        self.h_mask = h_mask
        self.h2o = torch.nn.Linear(modelConfig.rnn_size, osize)

    def forward(self, input, hidden):
        i = self.i2h(input)
        h_effective = torch.mul(self.h_w, self.h_mask)
        h = torch.matmul(hidden, h_effective)
        hidden = torch.relu(i + h + self.h_b)
        out = self.h2o(hidden)
        return hidden, out

    def initialZeroState(self):
        return torch.zeros(self.batch_size, self.hidden_size)

class Constrained_Model(Abstract_Model):
    def __init__(self, opts, isize, osize):
        super(Constrained_Model, self).__init__(opts.save_path)

        self.hidden_size = opts.rnn_size
        self.batch_size = opts.batch_size
        self.config = opts

        input_position_size = isize - 2
        input_velocity_size = 2
        hidden_attractor_size = opts.state_size
        hidden_shift_size = opts.rnn_size - opts.state_size

        self.position2attractor = nn.Linear(input_position_size, hidden_attractor_size)
        self.velocity2shift = nn.Linear(input_velocity_size, hidden_shift_size)

        self.h_w = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size, opts.rnn_size))
        self.h_b = torch.nn.Parameter(.01 * torch.rand(opts.rnn_size))

        mask = np.ones((opts.rnn_size, opts.rnn_size)).astype(np.float32)
        np.fill_diagonal(mask, 0)
        mask[opts.state_size:, opts.state_size:] = 0
        mask = torch.from_numpy(mask)
        h_mask = torch.nn.Parameter(mask, requires_grad=False)
        self.h_mask = h_mask
        self.h2o = torch.nn.Linear(opts.rnn_size, osize)

    def forward(self, input, hidden):
        i_attractor = self.position2attractor(input[:,:-2])
        i_shift = self.velocity2shift(input[:,-2:])
        i = torch.cat((i_attractor, i_shift), dim=1)

        h_effective = torch.mul(self.h_w, self.h_mask)
        h = torch.matmul(hidden, h_effective)
        hidden = torch.relu(i + h + self.h_b)
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
        nE = int(opts.rnn_size * opts.prop_ex)
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
        hidden = torch.relu(i + h + self.h_b)

        h2o_effective = torch.matmul(self.ei_mask, torch.abs(self.h2o_w))
        out = torch.matmul(hidden, h2o_effective) + self.h2o_b
        return hidden, out

    def initialZeroState(self):
        return torch.zeros(self.batch_size, self.hidden_size)


def load_config(save_path, name, epoch=None):
    if epoch is not None:
        save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))

    with open(os.path.join(save_path, name), 'r') as f:
        config_dict = json.load(f)

    c = config.modelConfig()
    for key, val in config_dict.items():
        setattr(c, key, val)
    return c