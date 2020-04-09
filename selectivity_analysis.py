import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
from scipy.stats import ranksums
import os
import pickle as pkl
import analysis_helper
import config


def get_selectivity_data(data_dict, ix_dict, opts):
    phase_ix, trial_type = data_dict['task_phase_ix'], data_dict['trial_type']
    if opts.mode[:3] == 'XJW':
        h = data_dict['r']  # use the firing rates, not current
    else:
        h = data_dict['h']
    return h, phase_ix, trial_type, ix_dict['active']


def test_selectivity(data_dict, ix_dict, opts, cutoff=.05):
    h, phase_ix, trial_type, active = get_selectivity_data(data_dict, ix_dict, opts)
    test_mean = np.mean(h[:, phase_ix['test']:phase_ix['response'],:], axis=1)  # N x D

    atest = (trial_type == 0) & (trial_type == 3)
    btest = ~atest
    a_test_means = test_mean[atest]
    b_test_means = test_mean[btest]

    a_test_selective, b_test_selective, test_stat, test_pval = \
        determine_ranksum_selectivity(a_test_means, b_test_means, alpha=.05)

    phase_active = np.amax(test_mean, axis=0) > cutoff
    active = active & phase_active

    return a_test_selective & active, b_test_selective & active, test_stat, test_pval


def sample_selectivity(data_dict, ix_dict, opts, cutoff=.05):
    h, phase_ix, trial_type, active = get_selectivity_data(data_dict, ix_dict, opts)
    sample_mean = np.mean(h[:, phase_ix['sample']:phase_ix['delay'], :], axis=1)  # N x D

    asample = (trial_type == 0) & (trial_type == 1)
    bsample = ~asample
    a_sample_means = sample_mean[asample]
    b_sample_means = sample_mean[bsample]

    a_sample_selective, b_sample_selective, sample_stat, sample_pval = \
        determine_ranksum_selectivity(a_sample_means, b_sample_means, alpha=.05)

    phase_active = np.amax(sample_mean, axis=0) > cutoff
    active = active & phase_active

    return a_sample_selective & active, b_sample_selective & active, sample_stat, sample_pval


def memory_selectivity(data_dict, ix_dict, opts, cutoff=.05):
    h, phase_ix, trial_type, active = get_selectivity_data(data_dict, ix_dict, opts)
    memory_activity = h[:, int(phase_ix['test'] - .5 / opts.dt):phase_ix['test'], :]
    mean = np.mean(memory_activity, axis=1)  # N x D

    amemory = (trial_type == 0) & (trial_type == 1)
    bmemory = ~amemory
    a_memory_means = mean[amemory]
    b_memory_means = mean[bmemory]

    a_memory_selective, b_memory_selective, stat, pval = \
        determine_ranksum_selectivity(a_memory_means, b_memory_means, alpha=.05)

    phase_active = np.amax(mean, axis=0) > cutoff
    active = active & phase_active

    return a_memory_selective & active, b_memory_selective & active, stat, pval


def choice_selectivity(data_dict, ix_dict, opts, cutoff=.05):
    h, phase_ix, trial_type, active = get_selectivity_data(data_dict, ix_dict, opts)
    choice_mean = np.mean(h[:, phase_ix['response']:, :], axis=1)  # N x D

    match = (trial_type == 0) & (trial_type == 2)
    nonmatch = ~match
    match_means = choice_mean[match]
    nonmatch_means = choice_mean[nonmatch]

    match_selective, nonmatch_selective, stat, pval = \
        determine_ranksum_selectivity(match_means, nonmatch_means, alpha=.05)

    phase_active = np.amax(choice_mean, axis=0) > cutoff
    active = active & phase_active

    return match_selective & active, nonmatch_selective & active, stat, pval


def trial_type_selectivity(a_test_selective, b_test_selective, data_dict, ix_dict, opts):
    h, phase_ix, trial_type, active = get_selectivity_data(data_dict, ix_dict, opts)
    test_mean = np.mean(h[:, phase_ix['test']:phase_ix['response'], :], axis=1)  # N x D
    tt_stat = np.zeros(opts.rnn_size)
    tt_pval = np.zeros(opts.rnn_size)

    aa_means = test_mean[trial_type == 0]
    ba_means = test_mean[trial_type == 3]
    _aa_selective, _ba_selective, stats, pvals = determine_ranksum_selectivity(aa_means, ba_means)
    aa_selective = _aa_selective & a_test_selective
    ba_selective = _ba_selective & a_test_selective

    tt_stat[aa_selective] = stats[aa_selective]
    tt_stat[ba_selective] = stats[ba_selective]
    tt_pval[aa_selective] = pvals[aa_selective]
    tt_pval[ba_selective] = pvals[ba_selective]

    ab_means = test_mean[trial_type == 1]
    bb_means = test_mean[trial_type == 2]
    _ab_selective, _bb_selective, stats, pvals = determine_ranksum_selectivity(ab_means, bb_means)
    ab_selective = _ab_selective & b_test_selective
    bb_selective = _bb_selective & b_test_selective

    tt_stat[ab_selective] = stats[ab_selective]
    tt_stat[bb_selective] = stats[bb_selective]
    tt_pval[ab_selective] = pvals[ab_selective]
    tt_pval[bb_selective] = pvals[bb_selective]

    return aa_selective, ab_selective, bb_selective, ba_selective, tt_stat, tt_pval


def determine_ranksum_selectivity(x, y, alpha=.05, abs_thresh=.05):
    """
    Determine whether there is a selectivity difference using a rank-sum test.
    :param x: (num_x_trials x rnn_size) array
    :param y: (num_y_trials x rnn_size) array
    Do I need to reimplement active-by-period?
    """
    stats, pvals = [], []
    for d in range(x.shape[1]):
        stat, p = ranksums(x[:, d], y[:, d])
        stats.append(stat)
        pvals.append(p)

    stats = np.array(stats)
    pvals = np.array(pvals)
    selective = pvals < alpha

    # determine which option a neuron is selective for
    _x_selective = np.mean(x, axis=0) > (np.mean(y, axis=0) + abs_thresh)
    _y_selective = np.mean(y, axis=0) > (np.mean(x, axis=0) + abs_thresh)
    x_selective = _x_selective & selective
    y_selective = _y_selective & selective
    return x_selective, y_selective, stats, pvals


def selectivity_analysis(opts):
    with open(os.path.join(save_path, 'analysis.pkl'), 'wb') as f:
        save_dict = pkl.load(f)

    data_dict = save_dict['data']
    weight_dict = save_dict['weights']
    ix_dict = save_dict['ix']

    # odor selectivity
    a_test_selective, b_test_selective, test_stat, test_pval = \
        test_selectivity(data_dict, ix_dict, opts)
    a_sample_selective, b_sample_selective, sample_stat, sample_pval = \
        sample_selectivity(data_dict, ix_dict, opts)
    odor_a_selective = a_sample_selective & a_test_selective
    odor_b_selective = b_sample_selective & b_test_selective
    a_memory_selective, b_memory_selective, memory_stat, memory_pval = \
        memory_selectivity(data_dict, ix_dict, opts)

    # match/nonmatch and trial type tuning
    match_selective, nonmatch_selective, choice_stat, choice_pval = \
        choice_selectivity(data_dict, ix_dict, opts)

    aa_selective, ab_selective, bb_selective, ba_selective, tt_stat, tt_pval = \
        trial_type_selectivity(a_test_selective, b_test_selective, data_dict, ix_dict, opts)

    odor = odor_a_selective & odor_b_selective
    trial_type = aa_selective & ab_selective & bb_selective & ba_selective
    choice = match_selective & nonmatch_selective
    memory = a_memory_selective & b_memory_selective
    sample = a_sample_selective & b_sample_selective
    test = a_test_selective & b_test_selective

    ix_dict = dict(a_test_selective=a_test_selective,
             b_test_selective=b_test_selective,
             a_sample_selective=a_sample_selective,
             b_sample_selective=b_sample_selective,
             odor_a_selective=odor_a_selective,
             odor_b_selective=odor_b_selective,
             aa_selective=aa_selective,
             ab_selective=ab_selective,
             bb_selective=bb_selective,
             ba_selective=ba_selective,
             match_selective=match_selective,
             nonmatch_selective=nonmatch_selective,
             a_memory_selective=a_memory_selective,
             b_memory_selective=b_memory_selective,
             active=ix_dict['active'],
             odor=odor,
             trial_type=trial_type,
             choice=choice,
             memory=memory,
             sample=sample,
             test=test)

    count_dict = dict()
    for k, v in ix_dict.items():
        count_dict[k] = np.sum(v)
        print(f'{v} {k} selective neurons')

    save_dict = dict(data=data_dict, weights=weight_dict, ix=ix_dict)
    with open(os.path.join(opts.save_path, 'analysis.pkl'), 'wb') as f:
        pkl.dump(save_dict, f)


def plot_selectivity(plot_path):
    with open(os.path.join(save_path, 'analysis.pkl'), 'wb') as f:
        save_dict = pkl.load(f)

    data_dict = save_dict['data']
    ix_dict = save_dict['ix']

    mean, sem = data_dict['mean'], data_dict['sem']
    color_dict, phase_ix = data_dict['color_dict'], data_dict['task_phase_ix']
    phase_ix = [phase_ix['sample'], phase_ix['delay'], phase_ix['test'], phase_ix['response']]

    categories = ['a_test_selective', 'b_test_selective', 'a_sample_selective', 'b_sample_selective',
     'odor_a_selective', 'odor_b_selective', 'aa_selective', 'ab_selective', 'bb_selective', 'ba_selective',
     'match_selective', 'nonmatch_selective', 'a_memory_selective', 'b_memory_selective', 'trial_type']

    for c in categories:
        me, se = [m[:, ix_dict[c]] for m in mean], [s[:, ix_dict[c]] for s in sem]
        analysis_helper.make_activity_plot(me, se, color_dict, phase_ix, plot_path, plot_name=c)


if __name__ == '__main__':
    root = './'
    if not os.path.exists(root + '_FIGURES'):
        os.mkdir(root + '_FIGURES')

    mode = 'XJW_EI'
    save_path = './_DATA/' + mode
    plot_path = './_FIGURES/' + mode
    opts = config.load_config(save_path, mode)
    opts.save_path = save_path
    selectivity_analysis(opts)
    plot_selectivity(plot_path)