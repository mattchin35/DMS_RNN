import os

from matplotlib import pyplot as plt
import matplotlib as mpl

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['font.family'] = 'arial'
fig_path = '/Users/peterwang/Desktop/PYTHON/Neural_head_direction/_FIGURES'

def _easy_save(path, name, dpi=300, pdf=True):
    '''
    convenience function for saving figs while taking care of making folders
    :param path: save path
    :param name: save name
    :param dpi: save dpi for .png format
    :param pdf: boolean, save in another pdf or not
    :return:
    '''
    path = os.path.join(fig_path, path)
    os.makedirs(path, exist_ok=True)
    figname = os.path.join(path, name)
    print('figure saved in {}'.format(figname))
    plt.savefig(os.path.join(figname + '.png'), dpi=dpi)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
    plt.close()

def _nice_figures(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out', length=1, width=.5)


def torch2numpy(x):
    return x.detach().cpu().numpy()