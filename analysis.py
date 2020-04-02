import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch_train
import torch_model
import os
import pickle as pkl
import json
import analysis_helper
from collections import defaultdict
from tools import torch2numpy
import utils


def get_weights(net, opts):
    weight_dict = defaultdict()
    for name, param in net.named_parameters():
        weight_dict[name] = torch2numpy(param)

    # ei_mask = weight_dict['ei_mask']
    # weight_dict['h_w'] = np.matmul(ei_mask, np.abs(weight_dict['h_w']))
    # weight_dict['i2h'] = np.abs(weight_dict['i2h'])
    # weight_dict['h2o_w'] = np.matmul(ei_mask, np.abs(weight_dict['h2o_w']))

    return weight_dict


def get_data(opts, eval):
    fname = os.path.join(opts.save_path, 'test_log.pkl')
    if eval or not os.path.exists(fname):
        logger = torch_train.evaluate(opts, log=True)
    else:
        with open(fname, 'rb') as f:
            logger = pkl.load(f)
    return logger


def plot_performance(data_dict, trial_type, plot_path):
    labels, output = data_dict['y'], data_dict['y_out']
    # output = sp.special.softmax(output, axis=2)

    # find a trial for each trial type
    ix = [np.nonzero(trial_type == i)[0][0] for i in range(4)]
    data = [(np.argmax(output[i], axis=1), np.argmax(labels[i], axis=1)) for i in ix]

    plot_name = 'performance'
    utils.subplot_easy(data, 4, plot_path, plot_name, subtitles=('AA', 'AB', 'BB', 'BA'),
                       tight_layout=True, linewidth=.5, hide_ticks=True, ylim=[-.1,2.1])


def plot_activity():
    pass


def analyze_simple_network(opts, plot_path, eval=False):
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    # paths = opts.save_path.split('/')[2:]
    # plot_path = '/'.join(['./_FIGURES'] + paths)

    opts, data_loader, net = torch_train._initialize(opts, reload=True, set_seed=False)
    if eval:
        torch_train.evaluate(opts, log=True)
    weight_dict = get_weights(net, opts)
    data = get_data(opts, eval)
    cumul_time = analysis_helper.cumulative_time_dict(opts)
    trial_type, trial_dict, color_dict = analysis_helper.determine_trial_type(data['x'], cumul_time)

    plot_performance(data, trial_type, plot_path)
    # plot_activity(data, opts, plot_path)


if __name__ == '__main__':
    save_path = './_DATA/one_layer'
    plot_path = './_FIGURES/one_layer'
    opts = torch_model.load_config(save_path)
    opts.save_path = save_path
    analyze_simple_network(opts, plot_path)