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

    if opts.mode == 'EI':
        ei_mask = weight_dict['ei_mask']
        weight_dict['h_w'] = np.matmul(ei_mask, np.abs(weight_dict['h_w']))
        weight_dict['i2h'] = np.abs(weight_dict['i2h'])
        weight_dict['h2o_w'] = np.matmul(ei_mask, np.abs(weight_dict['h2o_w']))

    return weight_dict


def get_data(opts, eval):
    fname = os.path.join(opts.save_path, 'test_log.pkl')
    if eval or not os.path.exists(fname):
        logger = torch_train.evaluate(opts, log=True)
    else:
        with open(fname, 'rb') as f:
            logger = pkl.load(f)
    return logger


def plot_performance(data_dict, plot_path):
    labels, output, trial_type = data_dict['y'], data_dict['y_out'], data_dict['trial_type']

    # find a trial for each trial type
    ix = [np.nonzero(trial_type == i)[0][0] for i in range(4)]
    data = [(np.argmax(output[i], axis=1), np.argmax(labels[i], axis=1)) for i in ix]

    plot_name = 'performance'
    utils.subplot_easy(data, 4, plot_path, plot_name, subtitles=('AA', 'AB', 'BB', 'BA'),
                       tight_layout=True, linewidth=.5, hide_ticks=True, ylim=[-.1,2.1])


def plot_activity(data_dict, plot_path):
    h, y_out, x, y = data_dict['h'], data_dict['y_out'], data_dict['x'], data_dict['y']
    print("Max activity:", np.amax(h))

    nr = np.ceil(np.sqrt(opts.rnn_size)).astype(np.int32)
    nc = np.ceil(opts.rnn_size / nr).astype(np.int32)
    ylim = [-.1, np.amax(h) + .1]

    # collect the average activity for each neuron for each trial type
    trial_type, color_dict, phase_ix = data_dict['trial_type'], data_dict['color_dict'], data_dict['task_phase_ix']
    phase_ix = [phase_ix['sample'], phase_ix['delay'], phase_ix['test'], phase_ix['response']]
    mean, sem = [], []
    for i in range(4):
        tt = h[trial_type == 0]
        mean.append(np.mean(tt, axis=0))
        sem.append(sp.stats.sem(tt, ddof=0, axis=0))

    f, ax = plt.subplots(nr, nc)
    ax = np.ravel(ax, order='C')
    for i in range(h.shape[-1]):
        for j, (tt_mean, tt_sem) in enumerate(zip(mean, sem)):
            m, se = tt_mean[:, i], tt_sem[:, i]
            ax[i].plot(m, lw=.3, color=color_dict[j])
            ax[i].fill_between(m, m-se, m+se, lw=.3, alpha=.5, color=color_dict[j])

        for p in phase_ix:
            ax[i].plot([p, p], ylim, linewidth=.3, color='k', linestyle='dashed')

        ax[i].set_ylim(ylim)
        ax[i].set_xlim(0, h.shape[1])
        if i != nc*(nr-1):
            utils.hide_axis_ticks(ax[i])
        else:
            ax[i].set_yticks([0, np.amax(h)])
            [spine.set_linewidth(0.3) for spine in ax[i].spines.values()]

    plt.suptitle('Neural Activity by Trial Type')
    plot_name = os.path.join(plot_path, f'neural activity')
    format = 'png'  # none, png or pdf
    f.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500, format=format)
    plt.close('all')


def analyze_simple_network(opts, plot_path, eval=False):
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    opts, data_loader, net = torch_train._initialize(opts, reload=True, set_seed=False)
    if eval:
        torch_train.evaluate(opts, log=True)

    weight_dict = get_weights(net, opts)
    data = get_data(opts, eval)
    task_phase_ix = analysis_helper.cumulative_time_dict(opts)
    trial_type, trial_ix_dict, color_dict = analysis_helper.determine_trial_type(data['x'], cumul_time)
    data['trial_type'] = trial_type
    data['trial_ix_dict'] = trial_ix_dict
    data['color_dict'] = color_dict
    data['task_phase_ix'] = task_phase_ix

    plot_performance(data, plot_path)
    plot_activity(data, plot_path)


if __name__ == '__main__':
    save_path = './_DATA/EI'
    plot_path = './_FIGURES/EI'
    opts = torch_model.load_config(save_path, 'EI')
    opts.save_path = save_path
    analyze_simple_network(opts, plot_path)