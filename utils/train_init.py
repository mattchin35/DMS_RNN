import torch
from numpy import random
import random
import torch_model
import advanced_model
import os
import pickle as pkl

import numpy as np
from datasets import inputs
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from utils.tools import torch2numpy


class InputDataset(Dataset):
    def __init__(self, opts):
        X, Y = inputs.create_inputs(opts)
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


def _initialize(opts, reload, set_seed, test=False):
    np.set_printoptions(precision=2)
    if set_seed:
        seed = opts.rng_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    if test:
        assert opts.test_batch_size <= opts.n_input
        opts.batch_size = opts.test_batch_size

    use_cuda = torch.cuda.is_available()
    if opts.ttype == 'float':
        ttype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    else:
        ttype = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
    torch.set_default_tensor_type(ttype)

    dataset = InputDataset(opts)
    data_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)
    if opts.mode == 'one_layer':
        net = torch_model.Simple_Model(opts=opts, isize=dataset.X.shape[-1], osize=dataset.Y.shape[-1])
    elif opts.mode == 'EI':
        net = torch_model.EI_Model(opts=opts, isize=dataset.X.shape[-1], osize=dataset.Y.shape[-1])
    elif opts.mode == 'XJW_simple':
        net = advanced_model.XJW_Simple(opts=opts, isize=dataset.X.shape[-1], osize=dataset.Y.shape[-1])
    elif opts.mode == 'XJW_EI':
        net = advanced_model.XJW_EI(opts=opts, isize=dataset.X.shape[-1], osize=dataset.Y.shape[-1])
    elif opts.mode == 'three_layer':
        net = torch_model.Three_Layer_Model(opts=opts, isize=dataset.X.shape[-1], osize=dataset.Y.shape[-1])

    net.model_config = opts

    if reload:
        net.load(name='net')
    print('[***Saving Variables***]')
    for name, param in net.named_parameters():
        if param.requires_grad:
            print('{0:20}: {1}'.format(name, param.data.shape))

    opts.time_loss_end = int(np.sum([v for v in opts.trial_time.values()]) / opts.dt)
    if opts.fixation:
        opts.time_loss_start = 5
    else:
        opts.time_loss_start = opts.time_loss_end - int(opts.trial_time['response'] / opts.dt)
    return opts, data_loader, net