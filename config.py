import os
import json


class BaseConfig(object):
    def __init__(self):
        self.rng_seed = 0
        self.model_name = 'model'
        self.weight_name = 'weight'
        self.activity_name = 'activity'
        self.parameter_name = 'parameters'
        self.image_folder = 'images'
        self.log_name = 'log'

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)


class inputConfig(BaseConfig):
    def __init__(self):
        super(inputConfig, self).__init__()
        self.cdab = False
        self.n_input = 1000

        self.trial_time = {'no_lick': 1, 'sample': .5, 'delay': 1.5, 'test': .5, 'response': 1}
        self.dt = .02
        self.decay = True
        self.activation_fn = 'relu'
        self.dt = .02
        self.fixation = True


class baseModelConfig(inputConfig):
    def __init__(self):
        super(baseModelConfig, self).__init__()
        self.weight_alpha = .1
        self.activity_alpha = .1

        self.tau = 0.1  # neuronal time constant
        self.input_noise = .01
        self.network_noise = 0.05  # recurrent noise on each timestep
        self.decay = True
        self.testing = False
        self.load_checkpoint = False

        self.learning_rate = .001
        self.batch_size = 20
        self.test_batch_size = 100
        self.epoch = 200
        self.time_loss_start = 5
        self.time_loss_end = 20

        self.reload = False  # load checkpoint, overrides load_weights
        self.save_path = './_DATA/'

        self.ttype = 'float'
        self.print_epoch_interval = 5
        self.save_epoch_interval = 100

        self.debug_weights = False
        self.clip_gradient = False


class oneLayerModelConfig(baseModelConfig):
    def __init__(self):
        super(oneLayerModelConfig, self).__init__()
        self.rnn_size = 100
        self.save_path = './_DATA/one_layer'
        self.mode = 'one_layer'


class XJWModelConfig(baseModelConfig):
    def __init__(self):
        super(XJWModelConfig, self).__init__()
        self.rnn_size = 100
        self.activity_alpha = .05
        self.vanishing_gradient_mult = 0

        self.save_path = './_DATA/XJW_simple'
        self.mode = 'XJW_simple'


class EIModelConfig(baseModelConfig):
    def __init__(self):
        super(EIModelConfig, self).__init__()
        self.rnn_size = 500
        self.epoch = 500
        self.percent_E = .8
        self.save_path = './_DATA/EI'
        self.mode = 'EI'


class XJW_EIConfig(baseModelConfig):
    def __init__(self):
        super(XJW_EIConfig, self).__init__()
        self.rnn_size = 500
        self.percent_E = .8
        self.activity_alpha = .05
        self.vanishing_gradient_mult = 0

        self.save_path = './_DATA/XJW_EI'
        self.mode = 'XJW_EI'


class threeLayerModelConfig(baseModelConfig):
    def __init__(self):
        super(threeLayerModelConfig, self).__init__()
        self.rnn_size = [30, 30, 30]
        self.trainable = [True, True, True]
        self.save_path = './_DATA/three_layer'
        self.mode = 'three_layer'


class constrainedModelConfig(baseModelConfig):
    def __init__(self):
        super(constrainedModelConfig, self).__init__()
        self.pir_size = 20
        self.alm_size = 20
        self.save_path = './_DATA/constrained'
        self.mode = 'constrained'


class interneuronModelConfig(threeLayerModelConfig):
    def __init__(self):
        super(interneuronModelConfig, self).__init__()
        self.nE = 40
        self.nI = 40


def load_config(save_path, mode, epoch=None):
    assert mode in ['one_layer', 'EI', 'XJW_simple', 'XJW_EI'], "Invalid mode"
    if epoch is not None:
        save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))

    with open(os.path.join(save_path, 'model_config.json'), 'r') as f:
        config_dict = json.load(f)

    if mode == 'one_layer':
        c = oneLayerModelConfig()
    elif mode == 'EI':
        c = EIModelConfig()
    elif mode == 'XJW_simple':
        c = XJWModelConfig()
    elif mode == 'XJW_EI':
        c = XJW_EIConfig()

    for key, val in config_dict.items():
        setattr(c, key, val)
    return c



