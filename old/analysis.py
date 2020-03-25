import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import ranksums, mannwhitneyu, f_oneway, sem
import utils, config
import os, re, tables
import config
import pickle as pkl
import inputs, train
import pandas as pd
import analysis_helper as helper


def plot_activity(opts, eval=True, data=None):
    save_path = helper.get_save_path(opts)
    plot_path = helper.folder(helper.get_image_path(opts), 'overall_activity')
    rnn_size = opts.rnn_size
    activity_name = opts.activity_name

    # generate a plotting dataset
    if eval:
        train.eval(opts, data)

    with open(os.path.join(save_path, activity_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)
    states, predictions, X, Y, N = data_dict['states'], data_dict['predictions'], \
                                   data_dict['X'], data_dict['Y'], data_dict['N']
    print("Max activity:", np.amax(states))
    action = np.argmax(predictions, axis=2)
    labels = np.argmax(Y, axis=2)

    # determine the type of each trial
    cumul_time = helper.cumulative_time_dict(opts)
    trial_type, trial_dict, color_dict = helper.determine_trial_type(X, cumul_time)

    # view the output of a example trials (plot performance)
    f_perf, ax_perf = plt.subplots(4, 1)
    # plt.suptitle('Trial responses')
    for i in range(4):
        # ax_perf[i].plot(predictions[i, cumul_time['response']:])
        ax_perf[i].plot(action[i, :], label='Action')
        ax_perf[i].plot(labels[i], label='Label')
        for k, v in cumul_time.items():
            ax_perf[i].plot([v, v], [-1, 2])
            plt.text(v, -.5, k, fontsize=8)#, color=color)

        ax_perf[i].set_title(trial_dict[trial_type[i]])
        ax_perf[i].set_ylim([-.5, 2.5])

    plt.legend()
    plt.tight_layout()
    plot_name = os.path.join(plot_path, f'performance')
    f_perf.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500)

    # plot the traces for each trial, then determine how to sort
    tup = []
    for i in range(states.shape[0]):
        tt = trial_dict[trial_type[i]]
        tup.append((f'{tt} activity', states[i]))

    plot_name = os.path.join(plot_path, f'activity')
    utils.subimage_easy(tup, 4, 1, save_name=plot_name)

    nr = np.ceil(np.sqrt(rnn_size)).astype(np.int32)
    nc = np.ceil(rnn_size / nr).astype(np.int32)
    ylim = [np.amin(states) - .1, np.amax(states) + .1]
    # for i in range(4):
    #     tup = [('', states[i,:,d]) for d in range(rnn_size)]
    #     plot_name = os.path.join(save_path, image_folder, f'{trial_dict[trial_type[i]]} activity')
    #     utils.subplot_easy(tup, nc, nr, save_name=plot_name, hide_ticks=True, ylim=[np.amin(states)-.1, np.amax(states)+.1])

    stages = list(cumul_time.values())

    r, c = 0, 0
    act = [states[:, :, d] for d in range(rnn_size)]
    f_neuron, ax_neuron = plt.subplots(nr, nc)
    # plot neuron activities separately
    for a in act:
        cur_ax = ax_neuron[r, c]
        for i, trial in enumerate(a):
            tt = trial_dict[trial_type[i]]
            cur_ax.plot(trial, linewidth=.3, color=color_dict[tt])

        [spine.set_linewidth(0.3) for spine in cur_ax.spines.values()]
        for s in stages:
            cur_ax.plot([s, s], ylim, linewidth=.3, color='k')
        helper.hide_ticks(cur_ax)
        cur_ax.set_ylim(ylim)
        cur_ax.set_xlim(0, states.shape[1])

        c += 1
        if c >= nc:
            r += 1
            c = 0

    plt.suptitle('Neural Activity by Trial Type')
    plot_name = os.path.join(plot_path, f'neural activity')
    format = 'png'  # none, png or pdf
    f_neuron.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500, format=format)

    # plot all activities for each trial
    ylim = [0, ylim[1]]
    for i in range(4):
        f_tt, ax_tt = plt.subplots()
        for d in range(rnn_size):
            ax_tt.plot(states[i,:,d], linewidth=.5)
        for s in stages:
            ax_tt.plot([s, s], ylim, linewidth=.5, color='k')
        ax_tt.set_ylim(ylim)
        ax_tt.set_xlim(0, states.shape[1])
        plt.suptitle(f'{trial_dict[trial_type[i]]} activity')
        plot_name = os.path.join(plot_path, f'{trial_dict[trial_type[i]]} activity.' + format)
        f_tt.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500, format=format)
    plt.close('all')

def plot_weights(opts):
    """Plot sample activity for the classes of data."""
    # Network parameters
    save_path = helper.get_save_path(opts)
    weight_name = 'weight'
    image_folder = opts.image_folder
    rnn_size = opts.rnn_size

    dfname = os.path.join(save_path, 'ix_dataframe.h5')
    df = pd.read_hdf(dfname, key='df')

    plot_path = os.path.join(save_path, image_folder, 'weights')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # activity_name = 'activity'  # four trials
    activity_name = 'test_data'  # like 200 trials
    with open(os.path.join(save_path, activity_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)
    states, predictions, X, Y, N = data_dict['states'], data_dict['predictions'], \
                                   data_dict['X'], data_dict['Y'], data_dict['N']

    # determine the type of each trial
    cumul_time = helper.cumulative_time_dict(opts)
    stages = list(cumul_time.values())
    trial_info = helper.determine_trial_type(X, cumul_time)
    trial_type, trial_dict, color_dict = trial_info

    AA = trial_type == 0
    AB = trial_type == 1
    BB = trial_type == 2
    BA = trial_type == 3
    trial_ix = ((AA, 0), (AB, 1), (BB, 2), (BA, 3))

    # load initial weights
    with open(os.path.join(save_path, 'init_weight.pkl'), 'rb') as f:
        init_weight_dict = pkl.load(f)
    Wh_init = init_weight_dict['model/hidden_weights:0'] * (1-np.eye(rnn_size))

    # load trained weights
    with open(os.path.join(save_path, 'weight.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    # for k in weight_dict.keys():
    #     print(k)

    Win = weight_dict['model/input_weights:0']
    Whh = weight_dict['model/hidden_weights:0'] * (1-np.eye(rnn_size))
    Wout = weight_dict['model/output_weights:0']
    print(np.sqrt(np.sum(Wh_init ** 2 - Whh)))

    tup = [('Initial Weights', Wh_init), ('Trained Weights', Whh)]
    plot_name = os.path.join(save_path, image_folder, 'weights_comparison')
    utils.subimage_easy(tup, 2, 1, save_name=plot_name, vmin=-.3, vmax=.3)

    cols = df.columns.values
    for c in cols:
        print(c, df[c].sum())

    active = df['active'].values
    for c in cols:
        df[c] = df[c] & df['active']
        cur_ix = np.flatnonzero(df[c].values)
        others = np.flatnonzero(~df[c].values)

        sort_ix = np.concatenate([cur_ix, others])
        Whh_sort = Whh[sort_ix, :][:, sort_ix]

        vlim = .5
        f = plt.figure()
        plt.imshow(Whh_sort, cmap='RdBu_r', vmin=-vlim, vmax=vlim, interpolation='none')
        format = 'pdf'  # none or png for png
        plot_name = os.path.join(plot_path, 'Wh_' + c)
        f.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500, format=format)
        plt.close('all')


# def analyze_selectivity(opts, plot=True, eval=False, load_df=True, data=None):
def analyze_selectivity(opts, eval=False):
    """Visualization of trained network."""
    # Network parameters
    save_path = os.path.join(os.getcwd(), 'training', opts.save_path[-8:])
    image_folder = opts.image_folder
    rnn_size = opts.rnn_size
    dt = opts.dt
    alpha = .01

    # create a test dataset
    prev_act_name = opts.activity_name
    prev_n_inputs = opts.n_inputs

    activity_name = 'test_data'
    if eval:
        opts.activity_name = activity_name
        opts.n_inputs = 200
        data = inputs.create_inputs(opts)
        train.eval(opts, data)
        opts.activity_name = prev_act_name
        opts.n_inputs = prev_n_inputs

    plot_path = os.path.join(save_path, image_folder)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    with open(os.path.join(save_path, activity_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)

    states, predictions, X, Y, N = data_dict['states'], data_dict['predictions'], \
                                   data_dict['X'], data_dict['Y'], data_dict['N']
    n_input, timesteps, _ = states.shape
    action = np.argmax(predictions, axis=2)
    labels = np.argmax(Y, axis=2)
    if np.sum(action[:, -1] == labels[:, -1]) < X.shape[0]:  # incorrect answer present
        print('network is not fully trained')
        return None

    # determine the type of each trial
    cumul_time = helper.cumulative_time_dict(opts)
    trial_type, trial_dict, color_dict = helper.determine_trial_type(X, cumul_time)

    AA = trial_type == 0
    AB = trial_type == 1
    BB = trial_type == 2
    BA = trial_type == 3
    trial_ix = (AA, AB, BB, BA)

    max_act = np.amax(states)
    neuron_max = np.amax(states, axis=(0,1))
    mean_max_act = np.mean(neuron_max)
    neuron_mean = np.mean(states, axis=(0,1))
    # perc_cutoff = .05
    cutoff = .05 * max_act
    active = (neuron_max > cutoff * max_act) > 0
    inactive = (1-active) > 0
    # cutoff = .1 * mean_max_act
    # active = (neuron_mean > cutoff) > 0

    # determine test odor selectivity
    # find the average activity of the trials during the test period
    test_activity = states[:, cumul_time['test']:cumul_time['response'],:]
    test_mean = np.mean(test_activity, axis=1)
    a_test_selective, b_test_selective, test_active, test_stat, test_pval = helper._test_selectivity(test_mean, trial_ix,
                                                                                            cutoff, alpha)

    # then determine if neuron is selective for same odor during sample
    sample_activity = states[:, cumul_time['sample']:cumul_time['delay'], :]
    sample_mean = np.mean(sample_activity, axis=1)
    a_sample_selective, b_sample_selective, sample_active, sample_stat, sample_pval = helper.sample_selectivity(sample_mean,
                                                                                                         trial_ix,
                                                                                                         cutoff,
                                                                                                         alpha)

    # odor selectivity
    odor_a_selective = (a_sample_selective * a_test_selective) > 0
    odor_b_selective = (b_sample_selective * b_test_selective) > 0

    # trial type selectivity
    aa_selective, ab_selective, bb_selective, ba_selective, tt_active, tt_stat, tt_pval = \
        helper.trial_selectivity(a_test_selective, b_test_selective, test_mean, trial_ix, cutoff, alpha)

    # choice selectivity
    choice_activity = states[:, cumul_time['response']:, :]
    choice_means = np.mean(choice_activity, axis=1)
    match_selective, nonmatch_selective, match_active, choice_stat, choice_pval = helper.choice_selectivity(choice_means, trial_ix, cutoff, alpha)

    # memory selectivity - check if sample selective at the end of delay, and make sure it was sample selective before?
    memory_activity = states[:, int(cumul_time['test']-.5/dt):cumul_time['test'], :]
    memory_mean = np.mean(memory_activity, axis=1)
    a_memory_selective, b_memory_selective, memory_active, choice_stat, choice_pval = helper.sample_selectivity(memory_mean, trial_ix, cutoff, alpha)
    # a_memory_selective = a_memory_selective * a_sample_selective
    # b_memory_selective = b_memory_selective * b_sample_selective

    active = (test_active + sample_active + tt_active + match_active + memory_active) > 0
    inactive = (1 - active) > 0

    # collect identified indices in pandas
    # odor = (a_test_selective + b_test_selective + a_sample_selective + b_sample_selective) > 0
    odor = (odor_a_selective + odor_b_selective) > 0
    trial_type = (aa_selective + ab_selective + bb_selective + ba_selective) > 0
    choice = (match_selective + nonmatch_selective) > 0
    memory = (a_memory_selective + b_memory_selective) > 0
    sample = (a_sample_selective + b_sample_selective) > 0
    test = (a_test_selective + b_test_selective) > 0

    d = dict(a_test_selective=a_test_selective,
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
             active=active,
             inactive=inactive,
             odor=odor,
             trial_type=trial_type,
             choice=choice,
             memory=memory,
             sample=sample,
             test=test)

    df = pd.DataFrame(d)
    dfname = os.path.join(save_path, 'ix_dataframe.h5')
    # dfname = os.path.join(save_path, 'selectivity_ix.h5')
    df.to_hdf(dfname, key='df', mode='w')

    for c in df.columns.values:
        if c != 'inactive':
            df[c] = df[c] & df['active']

    df = df.sum(axis=0)
    count_dict = {}
    for k,v in d.items():
        count_dict[k] = np.sum(v*active)
    count_dict['inactive'] = rnn_size - np.sum(active)
    df = pd.Series(count_dict)
    dfname = os.path.join(save_path, 'selectivity_counts.h5')
    df.to_hdf(dfname, key='df', mode='w')
    plt.close('all')
    return df


def plot_activity_classes(opts):
    """Plot sample activity for the classes of data."""
    # Network parameters
    save_path = helper.get_save_path(opts)
    # activity_name = opts.activity_name
    image_folder = opts.image_folder
    rnn_size = opts.rnn_size

    dfname = os.path.join(save_path, 'ix_dataframe.h5')
    df = pd.read_hdf(dfname, key='df')

    plot_path = os.path.join(save_path, image_folder)
    # if not os.path.exists(plot_path):
    #     os.makedirs(plot_path)

    # activity_name = 'activity'  # four trials
    activity_name = 'test_data'  # like 200 trials
    with open(os.path.join(save_path, activity_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)
    states, predictions, X, Y, N = data_dict['states'], data_dict['predictions'], \
                                   data_dict['X'], data_dict['Y'], data_dict['N']

    # determine the type of each trial
    cumul_time = helper.cumulative_time_dict(opts)
    stages = list(cumul_time.values())
    trial_info = helper.determine_trial_type(X, cumul_time)
    trial_type, trial_dict, color_dict = trial_info

    AA = trial_type == 0
    AB = trial_type == 1
    BB = trial_type == 2
    BA = trial_type == 3
    trial_ix = ((AA, 0), (AB, 1), (BB, 2), (BA, 3))
    # f, ax = plt.subplots()
    tr_means = []
    tr_sem = []
    tr_type = []
    for ix, tt in trial_ix:
        mean = np.mean(states[ix], axis=0)
        error = sem(states[ix], axis=0)

        tr_means.append(mean)
        tr_sem.append(error)
        tr_type.append(tt)

    tr_means = np.stack(tr_means, axis=0)
    tr_sem = np.stack(tr_sem, axis=0)
    trial_info = (tr_type, trial_dict, color_dict)

    ylim = [np.amin(states) - .1, np.amax(states) + .1]
    cols = df.columns.values
    active = df['active'].values
    any_selective = np.zeros(rnn_size)
    for c in cols:
        ix = df[c].values
        if c != 'inactive':
            ix = (ix * active) > 0
            any_selective += ix

        print(c, np.sum(ix))
        # _act = states[:, :, ix]
        _act = tr_means[:, :, ix]
        _ste = tr_sem[:, :, ix]
        helper.plot_activity_by_neuron(_act, _ste, stages, trial_info, opts,
                                       name=c, ylim=ylim, img_subfolder='neural_activity')

    any_selective = any_selective > 0
    print('active equals selective', np.sum(active == any_selective))
    print('dead', rnn_size - np.sum(active))
    print('selective', np.sum(any_selective))
    print('nonselective', rnn_size - np.sum(any_selective))


def find_model(op_dict):
    # load parameter files, search for appropriate model
    modeldir = os.path.join(os.getcwd(), 'training')
    exp = re.compile('model([0-9]+)')
    found = []
    with os.scandir(modeldir) as models:
        for mod in models:
            dir = os.path.join(modeldir, mod.name)
            if not os.listdir(dir):
                continue

            match = exp.match(mod.name)
            if match:
                fname = os.path.join(modeldir, mod.name, 'parameters')
                opts = utils.load_parameters(fname)
                m_dict = vars(opts)
                m_property_match = np.array([m_dict[k] == v for k,v in op_dict.items()])
                if np.prod(m_property_match) == 1:  # all true
                    print(f'Model {match.group(1)} located in ' + opts.save_path)
                    found.append(opts)
    return found


def table(df=None):
    root = './training/'
    if df is None:
        dfname = os.path.join(root, 'overall_selectivity_counts.h5')
        df = pd.read_hdf(dfname, key='df')
        # df = pd.read_pickle(dfname)
    oldcols = list(df.columns.values)
    rows = df.axes[0].tolist()
    means = df.mean(axis=1)
    std = df.std(axis=1)
    sem = df.sem(axis=1)
    df = pd.concat([df, means, std, sem], axis=1)
    newcols = oldcols + ['mean', 'std', 'sem']
    df.columns = newcols
    print(df)
    # df.to_hdf(dfname, key='df', mode='w')

    dfname = os.path.join(root, 'overall_selectivity_counts_stats.h5')
    df.to_pickle(dfname)
    # dfname = os.path.join(root, 'overall_selectivity_counts_stats.csv')
    # df.to_csv(dfname)
    f, ax = plt.subplots()
    ix = np.arange(means.values.shape[0])
    plt.bar(ix, means.values, yerr=sem)
    ax.set_xticks(ix)
    ax.set_xticklabels(rows, rotation='vertical')
    plt.show()
    format = 'png'
    f.savefig('selectivity_counts', bbox_inches='tight', figsize=(14, 10), dpi=500, format=format)
    # print(df)


if __name__ == '__main__':
    op_dict = dict()
    op_dict['cdab'] = False
    op_dict['losses'] = 'error'
    op_dict['rnn_size'] = 200

    root = './training/'

    # numbers of models to analyze
    ix = [45]
    # for i in ix:
    #     helper.describe_model(i, op_dict)
    dirs = [os.path.join(root, 'model' + str(n).zfill(3)) for n in ix]
    # table()
    for d in dirs:
        fname = os.path.join(d, 'parameters')
        opts = utils.load_parameters(fname)

        # plot_activity(opts, eval=True)
        analyze_selectivity(opts, eval=True)
    #
        # plot_activity(opts, eval=False)
    #     analyze_selectivity(opts, eval=False)
    #     plot_activity_classes(opts)
