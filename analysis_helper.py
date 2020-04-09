import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums
import os
from utils import utils
import copy


def cumulative_time_dict(opts):
    trial_time = copy.deepcopy(opts.trial_time)
    dt = opts.dt
    for k, v in trial_time.items():
        trial_time[k] = int(v / dt)
    _cumul_time = np.cumsum(list(trial_time.values()))
    cumul_time = {'sample': _cumul_time[0], 'delay': _cumul_time[1],
                  'test': _cumul_time[2], 'response': _cumul_time[3], 'end': _cumul_time[4]}
    return cumul_time


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


def make_activity_plot(mean, sem, color_dict, phase_ix, plot_path, plot_name='neural_activity'):
    T, D = mean[0].shape
    bnd = np.amax(mean) * .05
    ylim = [-bnd, np.amax(mean) + bnd]
    if D == 100:
        nr,  nc = 10, 10
    elif D == 500:
        nr, nc = 20, 25
    else:
        nr = np.ceil(np.sqrt(D)).astype(np.int32)
        nc = np.ceil(D / nr).astype(np.int32)

    f, ax = plt.subplots(nr, nc)
    ax = np.ravel(ax, order='C')
    labels = ('AA', 'AB', 'BB', 'BA')
    for i in range(D):
        for j, (tt_mean, tt_sem) in enumerate(zip(mean, sem)):
            m, se = tt_mean[:, i], tt_sem[:, i]
            ax[i].plot(m, lw=.3, color=color_dict[j])
            ax[i].fill_between(m, m - se, m + se, lw=.3, alpha=.5, color=color_dict[j], label=labels[j])

        for p in phase_ix:
            ax[i].plot([p, p], ylim, linewidth=.3, color='k', linestyle='dashed')

        ax[i].set_ylim(ylim)
        ax[i].set_xlim(0, T)
        if i != nc * (nr - 1):
            utils.hide_axis_ticks(ax[i])
        else:
            ax[i].set_yticks([0, ylim[1]])
            ax[i].tick_params(width=.3)
        [spine.set_linewidth(0.3) for spine in ax[i].spines.values()]
    ax[D-1].legend()

    for i in range(D, nr*nc):
        ax[i].axis('off')

    # plt.suptitle('Neural Activity by Trial Type')
    plot_name = os.path.join(plot_path, plot_name)
    format = 'pdf'  # none, png or pdf
    f.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500, format=format)
    plt.close('all')


# def EI_weight_dist(weight_dict):
#     hE = weight_dict['hE']
#     hI = weight_dict['hI']
#     hEI = weight_dict['hEI']
#     hIE = weight_dict['hIE']
#
#     plt.hist(hE)


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

