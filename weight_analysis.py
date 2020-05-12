import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import pickle as pkl
import analysis
import analysis_helper
import config


def plot_matrix(hw, E_ix, I_ix, labels, plot_path, vmin=-.5, vmax=.5):
    E, I = np.concatenate(E_ix), np.concatenate(I_ix)
    all_ix = np.concatenate([E,I])
    hw_sort = hw[all_ix,:][:,all_ix]

    f, ax = plt.subplots()
    plt.imshow(hw_sort, cmap='RdBu_r', vmin=vmin, vmax=vmax)

    ticks = np.cumsum([0] + [len(ix) for ix in E_ix] + [len(ix) for ix in I_ix])
    text_posn = [ticks[i] + (ticks[i+1] - ticks[i]) / 2 for i in range(len(ticks)-1)]
    plt.xticks(ticks[1:], labels=[])
    plt.yticks(ticks[1:], labels=[])
    ax.xaxis.tick_top()
    for p, s in zip(text_posn, labels * 2):
        plt.text(p-2, -5, s, fontsize=3)  # X
        plt.text(-15, p, s, fontsize=5)  # Y
    ax.axis('image')

    figname = os.path.join(plot_path, 'sorted_weights.png')
    f.savefig(figname, bbox_inches='tight', figsize=(14, 10), dpi=500, format='png')


def weight_analysis(opts, plot_path):
    with open(os.path.join(opts.save_path, 'analysis.pkl'), 'rb') as f:
        save_dict = pkl.load(f)

    data_dict = save_dict['data']
    weight_dict = save_dict['weights']
    ix_dict = save_dict['ix']

    E, I = ix_dict['E'], ix_dict['I']

    tm = ix_dict['test_match_selective']
    tnm = ix_dict['test_nonmatch_selective']
    m = ix_dict['match_selective']
    nm = ix_dict['nonmatch_selective']
    dAtB = ix_dict['dAtB']
    dBtA = ix_dict['dBtA']
    dA = ix_dict['dA']
    dB = ix_dict['dB']
    mem_ns = ix_dict['memory_nonselective']
    test_ns = ix_dict['test_nonselective']

    hw = weight_dict['h_w']

    order = [tm, m, nm, dAtB, dBtA]#, mem_ns, test_ns]
    labels = ['tm', 'm', 'nm', 'dAtB', 'dBtA']#, 'mem_ns', 'test_ns']
    E_ix = [np.argwhere(ix & E).ravel() for ix in order]
    I_ix = [np.argwhere(ix & I).ravel() for ix in order]
    plot_matrix(hw, E_ix, I_ix, labels, plot_path)


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
    weight_analysis(opts, plot_path)