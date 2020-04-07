import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch_train
import advanced_train
import os
import pickle as pkl
import analysis_helper
from collections import defaultdict
from utils.tools import torch2numpy
from utils import utils
import config
from utils.train_init import _initialize


def get_weights(net, opts):
    weight_dict = defaultdict()
    for name, param in net.named_parameters():
        weight_dict[name] = torch2numpy(param)

    weight_dict['h_w'] = weight_dict['h_w'] * weight_dict['h_mask']

    if opts.mode in ['EI', 'XJW_EI']:
        ei_mask = weight_dict['ei_mask']

        weight_dict['h_w'] = np.matmul(ei_mask, np.abs(weight_dict['h_w']))
        weight_dict['i2h'] = np.abs(weight_dict['i2h'])
        weight_dict['h2o_w'] = np.matmul(ei_mask, np.abs(weight_dict['h2o_w']))

    return weight_dict


def get_data(opts, eval):
    fname = os.path.join(opts.save_path, 'test_log.pkl')
    if eval or not os.path.exists(fname):
        if opts.mode[:3] == 'XJW':
            advanced_train.evaluate(opts, log=True)
        else:
            torch_train.evaluate(opts, log=True)

    with open(fname, 'rb') as f:
        logger = pkl.load(f)
    return logger


def plot_performance(data_dict, plot_path):
    labels, output, trial_type = data_dict['y'], data_dict['y_out'], data_dict['trial_type']

    # find a trial for each trial type
    ix = [np.nonzero(trial_type == i)[0][0] for i in range(4)]
    data = [(np.argmax(output[i], axis=1), np.argmax(labels[i], axis=1)) for i in ix]

    plot_name = 'performance'
    utils.subplot_easy(data, 1, plot_path, plot_name, subtitles=('AA', 'AB', 'BB', 'BA'),
                       tight_layout=True, linewidth=.5, hide_ticks=True, ylim=[-.1,2.1])


def plot_unsorted_weights(data_dict, ix_dict, plot_path):
    hw = data_dict['h_w']
    plot_name = 'weights_unsorted'
    utils.subimage_easy((hw), 1, plot_path, plot_name)

    active = ix_dict['active']
    hw_active = hw[active][:, active]
    plot_name = 'active_weights_unsorted'
    utils.subimage_easy((hw_active), 1, plot_path, plot_name)


def plot_activity(data_dict, ix_dict, plot_path):
    mean, sem = data_dict['mean'], data_dict['sem']
    color_dict, phase_ix = data_dict['color_dict'], data_dict['task_phase_ix']
    phase_ix = [phase_ix['sample'], phase_ix['delay'], phase_ix['test'], phase_ix['response']]
    analysis_helper.make_activity_plot(mean, sem, color_dict, phase_ix, plot_path, plot_name='neural_activity')

    active = ix_dict['active']
    active_mean, active_sem = [m[:,active] for m in mean], [s[:, active] for s in sem]
    inactive_mean, inactive_sem = [m[:,~active] for m in mean], [s[:, ~active] for s in sem]

    analysis_helper.make_activity_plot(active_mean, active_sem, color_dict,
                                       phase_ix, plot_path, plot_name='active_neural_activity')
    analysis_helper.make_activity_plot(inactive_mean, inactive_sem, color_dict,
                                       phase_ix, plot_path, plot_name='inactive_neural_activity')


def get_active_neurons(data_dict, thresh=.05):
    if opts.mode[:3] == 'XJW':
        h = data_dict['r']
    else:
        h = data_dict['h']
    print("Max activity:", np.amax(h))

    # collect the average activity for each neuron for each trial type
    trial_type = data_dict['trial_type']
    mean, nmax, sem = [], [], []
    for i in range(4):
        tt = h[trial_type == i]
        mean.append(np.mean(tt, axis=0))
        sem.append(sp.stats.sem(tt, ddof=0, axis=0))
        nmax.append(np.amax(mean[-1], axis=0))
    max_activity = np.amax(np.stack(nmax, axis=0), axis=0)
    active = max_activity >= thresh
    data_dict['mean'] = mean
    data_dict['sem'] = sem
    ix_dict = dict(active=active)
    return ix_dict, data_dict


def analyze_simple_network(opts, plot_path, eval=False):
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    opts, data_loader, net = _initialize(opts, reload=True, set_seed=False)
    weight_dict = get_weights(net, opts)
    data_dict = get_data(opts, eval)
    task_phase_ix = analysis_helper.cumulative_time_dict(opts)
    trial_type, trial_ix_dict, color_dict = analysis_helper.determine_trial_type(data_dict['x'], task_phase_ix)
    data_dict['trial_type'] = trial_type
    data_dict['trial_ix_dict'] = trial_ix_dict
    data_dict['color_dict'] = color_dict
    data_dict['task_phase_ix'] = task_phase_ix

    ix_dict, data_dict = get_active_neurons(data_dict)

    # plot_performance(data_dict, plot_path)
    plot_activity(data_dict, ix_dict, plot_path)
    plot_unsorted_weights(weight_dict, ix_dict, plot_path)


if __name__ == '__main__':
    root = './'
    if not os.path.exists(root + '_FIGURES'):
        os.mkdir(root + '_FIGURES')

    save_path = './_DATA/XJW_EI'
    plot_path = './_FIGURES/XJW_EI'
    opts = config.load_config(save_path, 'XJW_EI')
    opts.save_path = save_path
    analyze_simple_network(opts, plot_path)