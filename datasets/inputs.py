import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import config


def create_inputs(opts):
    odors = np.random.randint(2, size=(opts.n_input, 2))
    action = 1 - (odors[:,0] == odors[:,1])
    if opts.cdab:
        odors[:,0] += 2

    trial_structure = dict()
    trial_structure['no_lick'] = int(opts.trial_time['no_lick'] / opts.dt)
    epochs = ['no_lick', 'sample', 'delay', 'test', 'response']
    for i, e in enumerate(epochs[1:]):
        trial_structure[e] = int(opts.trial_time[e] / opts.dt) + trial_structure[epochs[i]]

    inputs = np.zeros((opts.n_input, trial_structure['response'], 2 + int(opts.cdab)))
    labels = np.zeros((opts.n_input, trial_structure['response'], 3))
    for i in range(len(inputs)):
        # print(i, trial_structure[0], trial_structure[1], odors[i,0])
        inputs[i, trial_structure['no_lick']:trial_structure['sample'], odors[i,0]] = 1
        inputs[i, trial_structure['delay']:trial_structure['test'], odors[i,1]] = 1

        labels[i, trial_structure['test']:, action[i]] = 1
        if opts.fixation:
            labels[i, :trial_structure['test'], 2] = 1

    return inputs.astype(np.float32), labels.astype(np.float32)


if __name__ == '__main__':
    opts = config.inputConfig()
    inputs, labels = create_inputs(opts)
    k = 2
    plt.imshow(inputs[k])
    plt.figure()
    plt.imshow(labels[k])
    plt.show()