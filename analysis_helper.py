import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums
import os
from utils import utils
import copy


def hide_ticks(cur_axis):
    cur_axis.set_xticklabels([])
    cur_axis.set_yticklabels([])
    cur_axis.set_xticks([])
    cur_axis.set_yticks([])


def cumulative_time_dict(opts):
    trial_time = copy.deepcopy(opts.trial_time)
    dt = opts.dt
    for k, v in trial_time.items():
        trial_time[k] = int(v / dt)
    _cumul_time = np.cumsum(list(trial_time.values()))
    cumul_time = {'sample': _cumul_time[0], 'delay': _cumul_time[1],
                  'test': _cumul_time[2], 'response': _cumul_time[3], 'end': _cumul_time[4]}
    return cumul_time


def make_activity_plot(mean, sem, color_dict, phase_ix, plot_path, plot_name='neural_activity'):
    T, D = mean[0].shape
    ylim = [-.1, np.amax(mean) + .1]
    if D == 100:
        nr,  nc = 10, 10
    elif D == 500:
        nr, nc = 20, 25
    else:
        nr = np.ceil(np.sqrt(D)).astype(np.int32)
        nc = np.ceil(D / nr).astype(np.int32)

    f, ax = plt.subplots(nr, nc)
    ax = np.ravel(ax, order='C')
    for i in range(D):
        for j, (tt_mean, tt_sem) in enumerate(zip(mean, sem)):
            m, se = tt_mean[:, i], tt_sem[:, i]
            ax[i].plot(m, lw=.3, color=color_dict[j])
            ax[i].fill_between(m, m - se, m + se, lw=.3, alpha=.5, color=color_dict[j])

        for p in phase_ix:
            ax[i].plot([p, p], ylim, linewidth=.3, color='k', linestyle='dashed')

        ax[i].set_ylim(ylim)
        ax[i].set_xlim(0, D)
        if i != nc * (nr - 1):
            utils.hide_axis_ticks(ax[i])
        else:
            ax[i].set_yticks([0, ylim[1]])
            ax[i].tick_params(width=.3)
        [spine.set_linewidth(0.3) for spine in ax[i].spines.values()]

    plt.suptitle('Neural Activity by Trial Type')
    plot_name = os.path.join(plot_path, plot_name)
    format = 'png'  # none, png or pdf
    f.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500, format=format)
    plt.close('all')


def determine_trial_type(X, cumul_time):
    sample = X[:,cumul_time['sample'],:]
    test = X[:,cumul_time['test'],:]
    nonmatch = ~(sample == test)[:, 0]

    trial_type = np.zeros(X.shape[0])
    trial_type[sample[:,1] == 1] = 2
    trial_type += nonmatch
    trial_dict = {0: 'AA', 1: 'AB', 2: 'BB', 3: 'BA'}
    color_dict = {0: 'cornflowerblue', 1: 'blue', 2: 'lightsalmon', 3: 'red'}
    return trial_type, trial_dict, color_dict


def determine_selectivity(x, y, cutoff, alpha=.05, active_by_period=True):
    """
    Determine whether there is a selectivity difference using a rank-sum test.
    :param x: (num_x_trials x rnn_size) array
    :param y: (num_y_trials x rnn_size) array
    """
    test_stat, test_pval = [], []
    for d in range(x.shape[1]):
        stat, p = ranksums(x[:, d], y[:, d])
        # stat, p = mannwhitneyu(a_test_means[:, d], b_test_means[:, d], alternative='two-sided')  # equivalent to rank-sum
        test_stat.append(stat)
        test_pval.append(p)

    test_stat = np.array(test_stat)
    test_pval = np.array(test_pval)
    test_selective = test_pval < alpha
    # print(test_pval[test_selective])
    if active_by_period:
        active = ((np.amax(x, axis=0) > cutoff) + (np.amax(y, axis=0) > cutoff)) > 0
    else:
        active = None
    avg_act = (np.mean(x) + np.mean(y)) / 2
    # abs_difference = .1 * avg_act
    abs_difference = .05

    # determine which odor a neuron is selective for
    _x_selective = np.mean(x, axis=0) > np.mean(y, axis=0) + abs_difference
    _y_selective = np.mean(y, axis=0) > np.mean(x, axis=0) + abs_difference
    x_selective = (_x_selective * test_selective) > 0
    y_selective = ((_y_selective) * test_selective) > 0
    return x_selective, y_selective, active, test_stat, test_pval


def trial_selectivity(a_test_selective, b_test_selective, test_mean, trial_ix, cutoff,
                      alpha=.05, active_by_period=True):

    AA, AB, BB, BA = trial_ix
    rnn_size = test_mean.shape[1]
    aa_selective = np.zeros(rnn_size, dtype=np.int32)
    ab_selective = np.zeros(rnn_size, dtype=np.int32)
    bb_selective = np.zeros(rnn_size, dtype=np.int32)
    ba_selective = np.zeros(rnn_size, dtype=np.int32)
    tt_stat = np.zeros(rnn_size)
    tt_pval = np.zeros(rnn_size)

    if active_by_period:
        active = (np.amax(test_mean, axis=0) > cutoff) > 0
    else:
        active = None
    avg_act = np.mean(test_mean)
    print('test avg act', avg_act)
    # abs_difference = .1 * avg_act
    abs_difference = .05

    for i in range(rnn_size):
        if a_test_selective[i]:
            aa_means = test_mean[:, i][AA]
            ba_means = test_mean[:, i][BA]

            stat, p = ranksums(aa_means, ba_means)
            a_diff = np.mean(aa_means) - np.mean(ba_means)
            if p < alpha:
                # a_diff_norm = a_diff / (np.mean(aa_means) + np.mean(ba_means))
                if a_diff > abs_difference:
                    aa_selective[i] = 1
                elif np.abs(a_diff) > abs_difference:
                    ba_selective[i] = 1

                # if a_diff > 0:
                #     aa_selective[i] = 1
                # else:
                #     ba_selective[i] = 1

        elif b_test_selective[i]:
            bb_means = test_mean[:, i][BB]
            ab_means = test_mean[:, i][AB]

            b_diff = np.mean(bb_means) - np.mean(ab_means)
            stat, p = ranksums(bb_means, ab_means)
            # b_diff_norm = a_diff / (np.mean(bb_means) + np.mean(ab_means))
            if p < alpha:
                if b_diff > abs_difference:
                    bb_selective[i] = 1
                elif np.abs(b_diff) > abs_difference:
                    ab_selective[i] = 1

                # if b_diff > 0:
                #     bb_selective[i] = 1
                # else:
                #     ab_selective[i] = 1

        else:
            stat, p = np.nan, np.nan

        tt_stat[i] = stat
        tt_pval[i] = p

    return aa_selective > 0, ab_selective > 0, bb_selective > 0, ba_selective > 0, active, tt_stat, tt_pval


def choice_selectivity(choice_mean, trial_ix, cutoff, alpha=.05):
    AA, AB, BB, BA = trial_ix
    match = (AA + BB) > 0  # left choice trials
    match_means = choice_mean[match]
    nonmatch = (AB + BA) > 0  # right choice trials
    nonmatch_means = choice_mean[nonmatch]
    match_selective, nonmatch_selective, active, choice_stat, choice_pval = \
        determine_selectivity(match_means, nonmatch_means, cutoff, alpha=alpha)
    return match_selective, nonmatch_selective, active, choice_stat, choice_pval


def sample_selectivity(sample_mean, trial_ix, cutoff, alpha=.05):
    AA, AB, BB, BA = trial_ix
    a_sample = (AA + AB) > 0
    a_sample_means = sample_mean[a_sample]

    b_sample = (BA + BB) > 0
    b_sample_means = sample_mean[b_sample]

    a_sample_selective, b_sample_selective, active, sample_stat, sample_pval = determine_selectivity(a_sample_means,
                                                                                             b_sample_means,
                                                                                             cutoff,
                                                                                             alpha)
    return a_sample_selective, b_sample_selective, active, sample_stat, sample_pval


def _test_selectivity(test_mean, trial_ix, cutoff, alpha=.05):
    # use the test means to do a rank-sum test of test selectivity
    # group A tests (AA/BA) and B tests (BB/AB)
    AA, AB, BB, BA = trial_ix
    a_test = (AA + BA) > 0
    a_test_means = test_mean[a_test]

    b_test = (AB + BB) > 0
    b_test_means = test_mean[b_test]

    a_test_selective, b_test_selective, active, test_stat, test_pval = determine_selectivity(a_test_means, b_test_means,
                                                                                     cutoff, alpha)
    return a_test_selective, b_test_selective, active, test_stat, test_pval


def plot_activity_by_neuron(states, ste, stages, trial_info, opts, name='', ylim=None, img_subfolder=None):
    """
    :param states: actvities to be plotted. (N trials x T timesteps x D neurons)
    :param ste: standard error for each neuron for each trial. (N trials x T timesteps x D neurons)
    :param stages: start points for each stage of the trial
    :param trial_info: stuff to use for indexing and color options
    :param opts: same as everywhere else
    :param name: plot name to be used for saving figure.
    :param ylim: universal limits of all plots.
    :return: neithin
    """
    plot_path = folder(get_image_path(opts), img_subfolder)
    trial_type, trial_dict, color_dict = trial_info
    N, T, D = states.shape
    if D == 0:
        # print(f'No cells found.')
        return

    if ylim is None:
        ylim = [np.amin(states) - .1, np.amax(states) + .1]

    # act = [states[:, :, d] for d in range(D)]
    # ste = [ste[:, :, d] for d in range(D)]

    r, c = 0, 0
    nr = np.ceil(np.sqrt(D)).astype(np.int32)
    nc = np.ceil(D / nr).astype(np.int32)
    f_neuron, ax_neuron = plt.subplots(nr, nc)

    def get_ax():
        if D == 1:
            ax = ax_neuron
        else:
            ax = ax_neuron[c]
        return ax

    # plot neuron activities separately
    # for a, e in zip(act, ste):
    for d in range(D):
        a = states[:,:,d]
        e = ste[:,:,d]
        if nc > 1:
            try:
                cur_ax = ax_neuron[r, c]
            except IndexError:
                pass
        else:
            if D == 1:
                cur_ax = ax_neuron
            else:
                cur_ax = ax_neuron[c]

        for i, (trial, error) in enumerate(zip(a, e)):
            tt = trial_dict[trial_type[i]]
            cur_ax.plot(trial, linewidth=.3, color=color_dict[tt], label=tt)
            if ste is not None:
                cur_ax.fill_between(np.arange(trial.shape[0]), trial+error, trial-error, color=color_dict[tt], alpha=.5)

        [spine.set_linewidth(0.3) for spine in cur_ax.spines.values()]
        for s in stages:
            cur_ax.plot([s, s], ylim, linewidth=.3, color='k')

        cur_ax.set_ylim(ylim)
        cur_ax.set_xlim(0, states.shape[1])
        hide_ticks(cur_ax)

        c += 1
        if c >= nc:
            r += 1
            c = 0

    cur_ax.legend()
    # plt.suptitle('Neural Activity by Trial Type')
    if nc > 1:
        cur_ax = ax_neuron[nr-1, 0]
    else:
        if D == 1:
            cur_ax = ax_neuron
        else:
            cur_ax = ax_neuron[c]
    plt.sca(cur_ax)
    ylim = np.round(ylim, 2)
    plt.yticks(ylim, labels=ylim)

    # format = 'pdf'  # none or png for png
    # format = None  # none or png for png
    plot_name = os.path.join(plot_path, name + '.pdf')
    f_neuron.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500, format='pdf')
    plt.close('all')


def folder(save_path, new_folder=None):
    if new_folder:
        save_path = os.path.join(save_path, new_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    return save_path


def get_save_path(opts):
    return os.path.join(os.getcwd(), 'training', opts.save_path[-8:])


def get_image_path(opts):
    return os.path.join(get_save_path(opts), opts.image_folder)


def describe_model(model_ix, op_dict):
    modeldir = os.path.join(os.getcwd(), 'training')
    fname = os.path.join(modeldir, 'model' + str(model_ix).zfill(3), 'parameters')
    opts = utils.load_parameters(fname)
    m_dict = vars(opts)
    print(f"\nix: {model_ix}, Model attributes:")
    for k in op_dict.keys():
        print(f'{k}: {m_dict[k]}')