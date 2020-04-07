import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import matplotlib.cm as cm
import pickle
import itertools

def save_parameters(opts, save_name):
    cur_dict = opts.__dict__
    cur_dict = {k: v for k, v in cur_dict.items() if k[:2] != '__'}

    # save_name = save_name + '_' + opts.losses
    with open(save_name + '.json', 'w') as f:
        json.dump(cur_dict, f)

def load_parameters(save_path):
    """Load config."""
    with open(os.path.join(save_path + '.json'), 'r') as f:
        config_dict = json.load(f)

    class Config():
        pass

    config = Config()
    for key, val in config_dict.items():
        setattr(config, key, val)
    return config

def load_results(root_dir, subfile=None):
    dir = os.path.join(root_dir, 'files')
    dirs = [os.path.join(root_dir, dir, n) for n in os.listdir(dir)]
    dirs = sorted(dirs)
    mse_loss = []
    loss = []
    config = []
    for i, d in enumerate(dirs):
        if subfile is not None:
            d = os.path.join(d,subfile)
        config.append(load_parameters(os.path.join(d, 'parameters')))

        log_name = os.path.join(d, 'log.pkl')
        with open(log_name, 'rb') as f:
            log = pickle.load(f)
        mse_loss.append(log['mse_loss'])
        loss.append(log['loss'])
    return xe_loss, loss, config

def hide_axis_ticks(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

def subimage_easy(data, ncols, path, name, subtitles = [], suptitle = '',
                  cbar=False, vmin=-1, vmax=1, tight_layout=False, order='C',
                  colorbar=False, ax_op=['off', 'image'], pdf=False):
    """Make a figure with multiple subplots."""
    if subtitles:
        assert len(subtitles) == len(data), "Unequal lengths of subtitles and data lists"

    n_subplots = len(data)
    nrows = np.ceil(n_subplots / ncols).astype(int)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    if n_subplots > 1:
        ax = np.ravel(ax, order=order)
    else:
        ax = [ax]

    for i, (cur_axis, d) in enumerate(zip(ax, data)):
        if len(d.shape) < 2:
            d = np.expand_dims(d, 0)

        plt.sca(cur_axis)
        # sns.heatmap(w, cmap='RdBu_r', ax=ax[r, c], cbar=cbar, center=0)
        # sns.heatmap(d, cmap='RdBu_r', vmin=vmin, vmax=vmax, ax=cur_axis, cbar=cbar, center=0)
        plt.imshow(d, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        if colorbar:
            cb = plt.colorbar()
            cb.set_ticks([vmin, 0, vmax])
        if i == 0:
            plt.ylabel('Time')
        # cur_axis.set_xticklabels([])
        # cur_axis.set_yticklabels([])
        cur_axis.set_xticks([])
        cur_axis.set_yticks([0,49])

        if subtitles:
            cur_axis.set_title(subtitles[i])

        if ax_op is not None:
            for op in ax_op:
                cur_axis.axis(op)

    # cb = plt.colorbar()
    # cb.set_ticks([vmin, 0, vmax])

    if suptitle:
        plt.suptitle(suptitle)
    if tight_layout:
        plt.tight_layout()
    if not os.path.exists(path):
        os.mkdir(path)
    figname = os.path.join(path, name)
    fig.savefig(figname + '.png', bbox_inches='tight', figsize=(3, 2), dpi=500)
    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
    plt.close('all')

def subplot_easy(data, ncols, path, name, subtitles=[], suptitle='', xlim=None, ylim=None,
                 hide_ticks=False, ax_op=[], tight_layout=False, order='C', pdf=False, linewidth=None):
    """
    Make a plot of subplots and a single title. Line plots only, handle scatter plots and
    complicated other things separately.
    :param data: a list of each piece of data to be plotted.
    """
    if subtitles:
        assert len(subtitles) == len(data), "Unequal lengths of subtitles and data lists"

    n_subplots = len(data)
    nrows = np.ceil(n_subplots / ncols).astype(int)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    if n_subplots > 1:
        ax = np.ravel(ax, order=order)
    else:
        ax = [ax]

    for i, (cur_axis, d) in enumerate(zip(ax, data)):
        if isinstance(d, tuple) or isinstance(d, list):
            for _d in d:
                cur_axis.plot(_d, lw=linewidth)
        else:
            cur_axis.plot(d, lw=linewidth)

        if subtitles:
            cur_axis.set_title(subtitles[i])
        if xlim:
            cur_axis.set_xlim(xlim[0], xlim[1])
        if ylim:
            cur_axis.set_ylim(ylim[0], ylim[1])
        if hide_ticks:
            hide_axis_ticks(cur_axis)
        for op in ax_op:
            cur_axis.axis(op)

    if suptitle:
        plt.suptitle(suptitle)
    if tight_layout:
        plt.tight_layout()
    if not os.path.exists(path):
        os.mkdir(path)

    figname = os.path.join(path, name)
    fig.savefig(figname + '.png', bbox_inches='tight', figsize=(3, 2), dpi=500)
    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
    plt.close('all')

def adjust(ax):
    # plt.style.use('dark_background')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    # ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    # ax.yaxis.set_ticks([0, 0.5, 1.0])
    # ax.set_ylim([0, 1])

def plot_weights(data, save_path, ylabel ='', xlabel= ''):
    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes(rect)

    adjust(ax)
    vlim = np.round(np.max(abs(data)), decimals=1)
    im = ax.imshow(data, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(xlabel, labelpad=-5)
    ax.set_ylabel(ylabel, labelpad=-5)
    ax.set_xticks([0, data.shape[1]])
    ax.set_yticks([0, data.shape[0]])
    plt.axis('scaled')
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', labelpad=-10)
    plt.tick_params(axis='both', which='major')
    plt.savefig(os.path.join('./lab_meeting/images/' + save_path + '.png'),
                transparent=True)

def plot_weights_simple(data, save_path, ylabel='',xlabel='', vmin= None, vmax= None):
    print(plt.style.available)
    rect = [0.15, 0.15, 0.65, 0.65]
    # plt.style.use('seaborn-bright')
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes(rect)

    adjust(ax)
    if vmin is None or vmax is None:
        vmax = np.round(np.max(abs(data)), decimals=1)
        vmin = -vmax

    im = ax.imshow(data, cmap= 'RdBu_r', vmin=vmin, vmax=vmax,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(xlabel, labelpad=-5)
    ax.set_ylabel(ylabel, labelpad=-5)
    ax.set_xticks([0, data.shape[1]-1])
    ax.set_yticks([0, data.shape[0]-1])
    plt.tick_params(axis='both', which='major')
    # plt.axis('image')
    plt.savefig(save_path + '.png')


