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


class oneLayerModelConfig(inputConfig):

    def __init__(self):
        super(oneLayerModelConfig, self).__init__()
        self.rnn_size = 100
        self.weight_alpha = .1
        self.activity_alpha = .1

        self.tau = 0.1  # neuronal time constant
        self.noise = 0.05  # external noise on each timestep
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
        self.save_path = './_DATA/test'

        self.ttype = 'float'
        self.print_epoch_interval = 5
        self.save_epoch_interval = 100

        self.debug_weights = False


class threeLayerModelConfig(oneLayerModelConfig):
    def __init__(self):
        super(threeLayerModelConfig, self).__init__()
        self.rnn_size = [30, 30, 30]
        self.trainable = [True, True, True]


class constrainedModelConfig(oneLayerModelConfig):
    def __init__(self):
        super(constrainedModelConfig, self).__init__()
        self.pir_size = 20
        self.alm_size = 20


class EIModelConfig(oneLayerModelConfig):
    def __init__(self):
        super(EIModelConfig, self).__init__()
        self.percent_E = .8


class interneuronModelConfig(threeLayerModelConfig):

    def __init__(self):
        super(interneuronModelConfig, self).__init__()
        self.nE = 40
        self.nI = 40




