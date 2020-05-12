import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import pickle as pkl
import analysis
import analysis_helper
from collections import defaultdict
from utils.tools import torch2numpy
from utils import utils
import config
import copy


def prep_data(data_dict, opts):
    trial_type = data_dict['trial_type']
    r, h = data_dict['r'], data_dict['h']

    # find a trial for each trial type
    ix = [np.nonzero(trial_type == i)[0][0] for i in range(4)]
    h = np.array([h[i] for i in ix])
    r = np.array([r[i] for i in ix])
    return r, h, ix


def scatterplot_helper(data_dict, inputs, activity, ix_list, categories, plot_path, plot_name, suptitle=''):
    f, ax = plt.subplots(1, 2)
    plt.sca(ax[0])
    for ix, cat in zip(ix_list, categories):
        ix_cnt = np.sum(ix).astype(np.int32)
        x = np.concatenate([[i] * ix_cnt for i in range(len(inputs))])
        y = np.concatenate([ipt[ix] for ipt in inputs])
        plt.scatter(x, y, label=cat, s=.5)

    plt.xticks([0, 1, 2, 3], ['AA', 'AB', 'BB', 'BA'])
    plt.legend()

    plt.sca(ax[1])
    labels = ('AA', 'AB', 'BB', 'BA')
    color_dict, phase_ix_list = data_dict['color_dict'], data_dict['phase_ix_list']
    ylim = [np.amin(activity), np.amax(activity)]
    for j, m in enumerate(activity):
        plt.plot(m, color=color_dict[j], label=labels[j])

    for v in phase_ix_list:
        plt.plot([v, v], ylim, color='k', linestyle='dashed', linewidth=1)
    plt.legend()

    plt.ylim(ylim)
    plt.xlim([0, len(activity[0])])
    if suptitle:
        plt.suptitle(suptitle)
    plot_name = os.path.join(plot_path, plot_name)
    f.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500, format='pdf')
    plt.close('all')


def EI_input_scatterplot(inputs, activity, categories, data_dict, ix_dict, plot_path, plot_name, suptitle=''):
    E = ix_dict['active'] & ix_dict['E']
    I = ix_dict['active'] & ix_dict['I']
    print(categories)
    ix_list = [ix_dict[c] for c in categories]
    all_ix = copy.deepcopy(ix_list[0])
    for i in ix_list:
        all_ix = all_ix | i
    ix_list.append(~all_ix)
    E_ix_list = [ix & E for ix in ix_list]
    I_ix_list = [ix & I for ix in ix_list]
    new_cat = copy.deepcopy(categories)
    new_cat.append('null')

    scatterplot_helper(data_dict, inputs, activity, E_ix_list, new_cat, plot_path, plot_name+'_E', suptitle)
    # scatterplot_helper(data_dict, inputs, activity, I_ix_list, categories, plot_path, plot_name+'_I')


def EI_input_activity_plot(inputs, data_dict, remove_ix, ix_dict, plot_path, plot_name):
    mean, sem = data_dict['mean'], data_dict['sem']
    nE = np.sum(ix_dict['E'])

    ipt_sums = [np.sum(inputs[:,:nE], axis=1), np.sum(inputs[:,nE:], axis=1)]
    print('Esum',ipt_sums[0],'Isum',ipt_sums[1])

    Esort = np.flip(np.argsort(inputs[:, :nE], axis=1), axis=1)[:, :100]
    # E_ix = np.unique(np.concatenate([Esort[0], Esort[2]]))
    # E_ix = np.setdiff1d(Esort[2], remove_ix)
    E_ix = Esort[1]
    analysis_helper.make_activity_plot([m[:, E_ix] for m in mean], [s[:, E_ix] for s in sem],
                                       data_dict['color_dict'], data_dict['phase_ix_list'],
                                       plot_path, plot_name=plot_name + '_E_act')

    Isort = np.argsort(inputs[:, nE:], axis=1)[:, :60] + nE
    # I_ix = np.unique(np.concatenate([Isort[0], Isort[2]]))
    # I_ix = np.setdiff1d(Isort[2], remove_ix)
    I_ix = Isort[0]
    analysis_helper.make_activity_plot([m[:, I_ix] for m in mean], [s[:, I_ix] for s in sem],
                                       data_dict['color_dict'], data_dict['phase_ix_list'],
                                       plot_path, plot_name=plot_name + '_I_act')


def output_connections(data_dict, weight_dict, ix_dict, plot_path):
    h20, phase_ix_dict = weight_dict['h2o_w'], data_dict['phase_ix_dict']
    mean_out = data_dict['mean_out']  # [T x D] x4
    mean = data_dict['mean']
    h = np.stack(mean, axis=0)[:, phase_ix_dict['response']:, :]  # 4 x T x D

    plot_names = ['left_output', 'right_output']
    max_ix = []
    remove_ix = []
    for i in range(2):
        inputs = np.array([ht * h20[:, i] for ht in h])  # 4 x T x D
        inputs = np.mean(inputs, axis=1)  # average across time  # 4 x D
        categories = ['match_selective', 'nonmatch_selective']
        act = [m[:,i] for m in mean_out]
        EI_input_scatterplot(inputs, act, categories, data_dict, ix_dict, plot_path, plot_names[i],
                             suptitle='Inputs to Readouts')
        EI_input_activity_plot(inputs, data_dict, remove_ix, ix_dict, plot_path, plot_names[i])

        # take the most active neurons for left lick
        max_ix.append(np.flip(np.argsort(inputs, axis=1), axis=1)[:,:2])

    left_ix = np.unique(np.concatenate([max_ix[0][0], max_ix[0][2]]))
    right_ix = np.unique(np.concatenate([max_ix[1][1], max_ix[1][3]]))
    return left_ix, right_ix, max_ix


def match_connections(data_dict, weight_dict, ix_dict, plot_path, match=True):
    hw, phase_ix_dict = weight_dict['h_w'], data_dict['phase_ix_dict']
    mean = np.stack(data_dict['mean'], axis=0)
    h_st = mean[:, phase_ix_dict['test']:phase_ix_dict['test'] + int(.25 / data_dict['dt']), :]  # 4 x T x D
    h_fin = mean[:, phase_ix_dict['test'] + int(.25 / data_dict['dt']):phase_ix_dict['response'], :]  # 4 x T x D
    hmean = data_dict['hmean']

    if match:
        max_ix = np.argwhere(ix_dict['match_selective'])[0]
        plot_names = ['match']
    else:
        max_ix = np.argwhere(ix_dict['nonmatch_selective'])[3]
        plot_names = ['nonmatch']

    m, nm = ix_dict['match_selective'], ix_dict['nonmatch_selective']
    remove_ix = np.squeeze(np.argwhere(m | nm))
    for j, ix in enumerate(max_ix):
        categories = ['a_test_selective', 'b_test_selective']
        # categories = ['match_selective', 'test_match_selective',
        #               'nonmatch_selective', 'test_nonmatch_selective']#,
                      #'a_test_selective', 'b_test_selective']
        act = [m[:,ix] for m in hmean]

        inputs = np.array([h * hw[:, ix] for h in h_st])  # 4 x T x D
        inputs = np.mean(inputs, axis=1)  # average across time  # 4 x D
        EI_input_scatterplot(inputs, act, categories, data_dict, ix_dict, plot_path, plot_names[j] + '_st',
                             suptitle='Match neuron, starting test inputs')
        EI_input_activity_plot(inputs, data_dict, remove_ix, ix_dict, plot_path, plot_names[j] + '_st')

        inputs = np.array([h * hw[:, ix] for h in h_fin])  # 4 x T x D
        inputs = np.mean(inputs, axis=1)  # average across time  # 4 x D
        EI_input_scatterplot(inputs, act, categories, data_dict, ix_dict, plot_path, plot_names[j] + '_end',
                             suptitle='Match neuron, ending test inputs')
        EI_input_activity_plot(inputs, data_dict, remove_ix, ix_dict, plot_path, plot_names[j] + '_end')


def test_match_connections(data_dict, weight_dict, ix_dict, plot_path, match=True):
    hw, phase_ix_dict = weight_dict['h_w'], data_dict['phase_ix_dict']
    mean = np.stack(data_dict['mean'], axis=0)
    h_st = mean[:, phase_ix_dict['test']:phase_ix_dict['test'] + int(.25 / data_dict['dt']), :]  # 4 x T x D
    h_fin = mean[:, phase_ix_dict['test'] + int(.25 / data_dict['dt']):phase_ix_dict['response'], :]  # 4 x T x D
    mean, sem = data_dict['hmean'], data_dict['hsem']
    if match:
        max_ix = np.argwhere(ix_dict['test_match_selective'])[0]
        plot_names = ['test_match']
    else:
        max_ix = np.argwhere(ix_dict['test_nonmatch_selective'])[1]
        plot_names = ['test_nonmatch']

    tm, tnm = ix_dict['test_match_selective'], ix_dict['test_nonmatch_selective']
    m, nm = ix_dict['match_selective'], ix_dict['nonmatch_selective']
    remove_ix = np.squeeze(np.argwhere(tm | tnm | m | nm | ~ix_dict['active']))
    for j, ix in enumerate(max_ix):
        categories = ['a_test_selective', 'b_test_selective']
        # categories = ['match_selective', 'test_match_selective',
        #               'nonmatch_selective', 'test_nonmatch_selective']#,
                      #'a_test_selective', 'b_test_selective']
        act = [m[:,ix] for m in mean]

        inputs = np.array([h * hw[:, ix] for h in h_st])  # 4 x T x D
        inputs = np.mean(inputs, axis=1)  # average across time  # 4 x D
        EI_input_scatterplot(inputs, act, categories, data_dict, ix_dict, plot_path, plot_names[j] + '_st',
                             suptitle='Match neuron, starting test inputs')
        EI_input_activity_plot(inputs, data_dict, remove_ix, ix_dict, plot_path, plot_names[j] + '_st')

        inputs = np.array([h * hw[:, ix] for h in h_fin])  # 4 x T x D
        inputs = np.mean(inputs, axis=1)  # average across time  # 4 x D
        EI_input_scatterplot(inputs, act, categories, data_dict, ix_dict, plot_path, plot_names[j] + '_end',
                             suptitle='Match neuron, ending test inputs')
        EI_input_activity_plot(inputs, data_dict, remove_ix, ix_dict, plot_path, plot_names[j] + '_end')


def DT_connections(h, data_dict, weight_dict, ix_dict, plot_path):
    # get connections for neurons with delay selectivity for one odor and test selectivity for the other.
    hw, phase_ix_dict = weight_dict['h_w'], data_dict['phase_ix_dict']
    mean = np.stack(data_dict['mean'], axis=0)
    h = mean[:, phase_ix_dict['test']:phase_ix_dict['test'] + int(.25 / data_dict['dt']), :]  # 4 x T x D
    mean = data_dict['hmean']

    dA, tB = ix_dict['a_memory_selective'], ix_dict['b_test_selective']
    dAtB = dA & tB
    dB, tA = ix_dict['b_memory_selective'], ix_dict['a_test_selective']
    dBtA = dB & tA
    dAtA = dA & tA
    dBtB = dB & tB

    # plot_names = ['dAtB', 'dBtA']
    # max_ix = [np.squeeze(np.argwhere(dAtB))[0], np.squeeze(np.argwhere(dBtA))[0]]
    plot_names = ['dAtA', 'dBtB']
    max_ix = [np.squeeze(np.argwhere(dAtA))[0], np.squeeze(np.argwhere(dAtA))[0]]
    remove_ix = np.squeeze(np.argwhere(ix_dict['inactive']))

    # remove_ix = np.squeeze(np.argwhere(tm | tnm | m | nm | ~ix_dict['active']))

    for j, ix in enumerate(max_ix):
        inputs = np.array([ht * np.squeeze(hw[:, ix]) for ht in h])  # 4 x T x D
        inputs = np.mean(inputs, axis=1)  # average across time  # 4 x D
        act = [m[:, ix] for m in mean]
        # categories = ['aa_selective', 'ab_selective', 'bb_selective', 'ba_selective']
        categories = ['odor_a_selective', 'odor_b_selective']
        EI_input_scatterplot(inputs, act, categories, data_dict, ix_dict, plot_path, plot_names[j])
        EI_input_activity_plot(inputs, data_dict, remove_ix, ix_dict, plot_path, plot_names[j])


def connectivity_analysis(opts, plot_path):
    with open(os.path.join(opts.save_path, 'analysis.pkl'), 'rb') as f:
        save_dict = pkl.load(f)

    data_dict = save_dict['data']
    weight_dict = save_dict['weights']
    ix_dict = save_dict['ix']
    data_dict['dt'] = opts.dt

    # what are the biggest inputs to a given neuron or output? Start with output neurons
    # outputs are only a function of hidden activation at each step
    """
    - get the neural activity for a period (want test and end delay probably)
    - get the input to each neuron within that period
    - compare inputs: make a scatter plot
    """

    # output_connections(data_dict, weight_dict, ix_dict, plot_path)
    match_connections(data_dict, weight_dict, ix_dict, plot_path, match=False)
    # test_match_connections(r, h, data_dict, weight_dict, ix_dict, plot_path)
    # DT_connections(h, data_dict, weight_dict, ix_dict, plot_path)


if __name__ == '__main__':
    root = './'
    if not os.path.exists(root + '_FIGURES'):
        os.mkdir(root + '_FIGURES')

    mode = 'XJW_EI'
    save_path = os.path.join(root, '_DATA/', mode)
    plot_path = os.path.join(root, '_FIGURES/', mode, 'connectivity')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    opts = config.load_config(save_path, mode)
    opts.save_path = save_path
    connectivity_analysis(opts, plot_path)