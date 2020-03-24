import argparse
import config
import os, re
import copy
import numpy as np
import pickle as pkl
# import analysis

TORCH = False
if TORCH:
    from torch_train import train
else:
    from train import train
    # from train import train
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stationary_path', type=str,
                        default='./curriculum/stationary', help='stationary path')
    parser.add_argument('--nonstationary_path', type=str,
                        default='./curriculum/non_stationary', help='non-stationary path')
    return parser

def curriculum(params):
    # quick settings
    def fast_hd_trig(opts):
        opts.input_mode = 'trig'
        opts.output_mode = 'trig'
        opts.grid_input = False
        opts.linear_track = False
        return opts

    def fast_hd_bump(opts):
        opts.input_mode = 'bump'
        opts.output_mode = 'bump'
        opts.grid_input = False
        opts.linear_track = False
        return opts

    def fast_grid(opts):
        opts.input_mode = 'bump'
        opts.output_mode = 'bump'
        opts.grid_input = True
        opts.linear_track = True
        return opts

    rnn_size, state_size = params
    parser = arg_parser()
    opts = parser.parse_args()
    opts.rnn_size = rnn_size
    linear_track = False
    if linear_track:
        track = 'linear'
    else:
        track = 'circular'

    loss = 'error'  # ['error', 'activity', 'weights', 'full']
    # loss = 'activity'  # ['error', 'activity', 'weights', 'full']
    # loss = 'full'  # ['error', 'activity', 'weights', 'full']
    fn = 'relu'  # [relu, tanh]

    st_model_opts = config.stationary_model_config()
    # st_model_opts.output_mode = output_mode
    st_model_opts.weight_alpha = .5
    st_model_opts.activity_alpha = .1
    st_model_opts.rnn_size = rnn_size
    st_model_opts.state_size = state_size
    st_model_opts.linear_track = linear_track
    st_model_opts.losses = loss

    st_model_opts.activation_fn = fn
    st_model_opts.mask = True

    velocity_min = 1
    velocity_max = 3
    nonst_model_opts = config.non_stationary_model_config()
    # nonst_model_opts.output_mode = output_mode
    # nonst_model_opts.input_mode = input_mode
    nonst_model_opts.linear_track = linear_track
    nonst_model_opts.velocity_max = 3
    nonst_model_opts.weight_alpha = .5
    nonst_model_opts.activity_alpha = .1
    nonst_model_opts.rnn_size = rnn_size
    nonst_model_opts.state_size = state_size
    nonst_model_opts.losses = loss
    nonst_model_opts.activation_fn = fn
    nonst_model_opts.mask = True
    nonst_model_opts.boundary_velocity = False
    nonst_model_opts.correlated_path = False

    # get all present dirs
    modeldir = os.path.join(os.getcwd(), 'training')
    exp = re.compile('model([0-9]+)')
    used_ix = []
    with os.scandir(modeldir) as models:
        for m in models:
            rg = exp.match(m.name)
            if rg:
                used_ix.append(int(rg.group(1)))

    max_ix = np.amax(used_ix)
    full_ix = np.arange(max_ix+2)  # include the next index above the highest
    free_ix = [a for a in full_ix if a not in used_ix]
    new_ix = np.amin(free_ix)
    nonst_model_opts.save_path = os.path.join(os.getcwd(), 'training', f'model{str(new_ix).zfill(3)}')
    nonst_model_opts.dir_weights = os.path.join(nonst_model_opts.save_path, nonst_model_opts.model_name + '.pkl')

    # first = copy.deepcopy(st_model_opts)
    # # first.epoch = int(201)
    # first.epoch = int(1)
    # first.load_checkpoint = False
    #
    # first_more = copy.deepcopy(st_model_opts)
    # # first_more.epoch = int(101)
    # first_more.epoch = int(201)
    # first_more.learning_rate = 1e-4
    # first_more.load_checkpoint = True


    second = copy.deepcopy(nonst_model_opts)
    second.epoch = int(300)
    # second.epoch = int(1)
    second.load_checkpoint = False
    second.load_weights = False
    second.time_steps = 50
    second.time_loss_start = 5
    second.time_loss_end = second.time_steps
    second.velocity_step = 1
    second.velocity_min = velocity_min
    second.velocity_max = velocity_max
    second.dir_weights = os.path.join(st_model_opts.save_path,
                                      st_model_opts.weight_name + '.pkl')
    second.learning_rate = .001
    # second.debug_weights = True
    # n_env = 10
    # second.subtrack = False
    # second.rescale_env = True
    # second.n_env = n_env

    # second.bump_in_network = False
    # second.grid_input = False

    second.dropout = False
    second.dropout_rate = .4

    second.noise = False
    second.noise_intensity = .1
    second.noise_density = .5

    second.nonneg_input = True
    second.EI_in = False
    second.EI_h = False
    second.EI_out = False
    second.prop_ex = .8

    #### ASSIGN TRAINING PARADIGM ####
    second = fast_hd_trig(second)
    # second = fast_hd_bump(second)
    # second = fast_grid(second)

    # c = [first]
    # c = [first_more]
    # c = [first, first_more]

    c = [second]
    # c = [second_more]
    # c = [second, second_more]
    # c = [second_last]

    # c = [moving_static, second_more]
    return c


def subtrack_train(cur):
    """Training schedule for subtrack training. Run n_env times, then one more time with the full track."""
    assert cur[0].state_size >= cur[0].subtrack_maxlen, "Track length is longer than input size"
    op = cur[0]
    op.image_folder = 'image0'
    print("Round 0")
    run_track(op)

    op = cur[1]
    for env in range(op.n_env - 1):
        op.image_folder = f'image{env+1}'
        print(f"\nRound {env+1}")
        run_track(op)
    op.subtrack = False
    op.epoch = 50
    op.image_folder = 'imagefull'
    run_track(op)


def run_track(op, use_data=True):
    c, _ = train(op)
    train_path = os.path.join(op.save_path, 'training_set.pkl')
    if use_data:
        with open(train_path, 'rb') as f:
            data_dict = pkl.load(f)
    else:
        data_dict = None
    analysis.plot_activity(c, data=data_dict)
    analysis.analyze_nonstationary_weights(c, plot=True, eval=False, load_df=False, data=data_dict)


if __name__ == '__main__':
    # rnn_sizes = [25, 36, 50, 64, 100]
    rnn_sizes = [200]
    params = [(s, 36) for s in rnn_sizes]  # rnn size, input size
    for p in params:
        cur = curriculum(p)
        if cur[0].subtrack:
            subtrack_train(cur)
        else:
            for i, c in enumerate(cur):
                c, _ = train(c, seed=False)
                # c, _ = train(c, seed=True)
                print('[!] Curriculum %d has finished' % (i))
            # analysis.plot_activity(c)
            # analysis.analyze_nonstationary_weights(c, plot=True, eval=False, load_df=False)

