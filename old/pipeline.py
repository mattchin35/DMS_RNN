import train
from config import Options
import os, re, copy
import numpy as np
import pandas as pd
import analysis
import utils

def pipeline():
    opts = Options()

    # get all present dirs
    opts = get_model_ix(opts)

    # opts.trial_time = {'no_lick': .5, 'sample': .5, 'delay': 1.5, 'test': .5, 'response': 1}
    opts.trial_time = {'no_lick': .5, 'sample': .5, 'delay': 0, 'test': .5, 'response': .5}

    opts.fixation = True
    opts.mask = True

    opts = get_loss_times(opts)

    opts.activation_fn = 'relu'
    opts.decay = True
    opts.epoch = 100  # 100-200 is enough
    # opts.epoch = 10
    opts.batch_size = 20

    opts.multilayer = False
    opts.layer_size = [80, 80]

    opts.noise = False
    opts.noise_intensity = .01
    opts.noise_density = .5
    opts.learning_rate = .001

    opts.rnn_size = 100

    opts.weight_alpha = .1
    opts.activity_alpha = .1

    run0 = opts
    run1 = copy.deepcopy(opts)
    run1.trial_time['delay'] = .5
    run1 = get_loss_times(run1)
    run1.load_checkpoint = True
    run2 = copy.deepcopy(run1)
    run2.trial_time['delay'] = 1
    run2 = get_loss_times(run2)
    run3 = copy.deepcopy(run1)
    run3.trial_time['delay'] = 1.5
    run3 = get_loss_times(run3)

    for run in [run0,run1,run2,run3]:
        train.train(run)

    # opts.EI_in = False
    # opts.EI_h = False
    # opts.EI_out = False
    # opts.proportion_ex = .8
    # train.eval(opts)

    # run_multiple(opts)

    return opts

def run_multiple(opts):
    dfs = []
    k = 0
    while k < 20:
        opts.model_name += '_' + str(k).zfill(2)
        train.train(opts)
        # train.eval(opts)
        df = analysis.analyze_selectivity(opts, eval=True)
        # if df is None:
        #     continue
        # dfs.append(df)
        k += 1
    df = pd.concat(dfs, axis=1)
    print('concatencated df2\n', pd.concat(dfs, axis=1))
    dfname = os.path.join(opts.save_path, 'selectivity_counts.h5')
    df.to_hdf(dfname, key='df', mode='w')
    analysis.table(opts)

def collect_dist():
    ix = []

    k = 0
    # while k < 20:
    for k in range(20):
        opts = pipeline()
        ix.append(opts.save_path[-3:])
        # k += 1

    print('indices used:', ix)
    root = './training/'
    dirs = [os.path.join(root, 'model' + str(n).zfill(3)) for n in ix]

def analyze_dist():
    root = './training/'
    ix = np.arange(3, 23)
    dirs = [os.path.join(root, 'model' + str(n).zfill(3)) for n in ix]
    dfs = []
    for d in dirs:
        fname = os.path.join(d, 'parameters')
        opts = utils.load_parameters(fname)
        df = analysis.analyze_selectivity(opts, eval=False)
        if df is None:
            continue
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    # print('concatencated df2\n', pd.concat(dfs, axis=1))
    dfname = os.path.join(root, 'selectivity_counts.h5')
    # df.to_hdf(dfname, key='df', mode='w')
    df.to_pickle(dfname)
    analysis.table()

def get_model_ix(opts):
    modeldir = os.path.join(os.getcwd(), 'training')
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    exp = re.compile('model([0-9]+)')
    used_ix = []
    with os.scandir(modeldir) as models:
        for m in models:
            rg = exp.match(m.name)
            if rg:
                used_ix.append(int(rg.group(1)))

    if used_ix:
        max_ix = np.amax(used_ix)
    else:
        max_ix = -1
    full_ix = np.arange(max_ix + 2)  # include the next index above the highest
    free_ix = [a for a in full_ix if a not in used_ix]
    new_ix = np.amin(free_ix)
    opts.save_path = os.path.join(os.getcwd(), 'training', f'model{str(new_ix).zfill(3)}')
    opts.dir_weights = os.path.join(opts.save_path, opts.model_name + '.pkl')
    return opts

def get_loss_times(opts):
    trial_time = opts.trial_time.copy()
    for k, v in trial_time.items():
        trial_time[k] = int(v / opts.dt)
    _start_time = np.cumsum(list(trial_time.values()))
    start_time = {'sample': _start_time[0], 'delay': _start_time[1],
                  'test': _start_time[2], 'response': _start_time[3]}
    if opts.fixation:
        opts.time_loss_start = 0
    else:
        opts.time_loss_start = int(start_time['response'])
    opts.time_loss_end = int(start_time['response'] + trial_time['response'])
    return opts


if __name__ == '__main__':
    pipeline()
    # collect_dist()
    # analyze_dist()
    