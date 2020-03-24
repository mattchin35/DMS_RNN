import tensorflow as tf
from torch_model import Model


class RNN(Model):
    def __init__(self, data, opts, training=True):
        super(RNN, self).__init__(opts.save_path)

        X_pl, Y_pl, N_pl = data[0], data[1], data[2]
        self.opts = opts
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(X_pl, Y_pl, N_pl)

        if training:
            print('building optimizer...')
            learning_rate = opts.learning_rate
            optimizer = tf.train.AdamOptimizer(learning_rate)
            excludes = []
            trainable_list = [v for v in tf.trainable_variables() if v not in excludes]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.total_loss, var_list=trainable_list)

            print('[***] Training Variables:')
            for v in trainable_list:
                print(v)

        self.saver = tf.train.Saver()
        print('built')

    def _build(self, x, y, n):
        print('build start')
        opts = self.opts
        rnn_size = opts.rnn_size
        time_loss_start = opts.time_loss_start
        time_loss_end = opts.time_loss_end
        batch_size = opts.batch_size
        self.tau = opts.tau
        self.decay = opts.dt / opts.tau
        assert opts.activation_fn in ['relu', 'tanh', 'relu6', 'retanh', 'sigmoid'], "Invalid nonlinearity"

        fn = opts.activation_fn
        if fn == 'sigmoid':
            self.fn = tf.nn.sigmoid
        elif fn == 'tanh':
            self.fn = tf.tanh
        elif fn == 'relu':
            self.fn = tf.nn.relu
        elif fn == 'relu6':
            self.fn = tf.nn.relu6
        else:
            self.fn = lambda L: tf.nn.relu(tf.nn.tanh(L))  # a rectified tanh function

        inputs_series = tf.unstack(x, axis=1)
        labels_series = tf.unstack(y, axis=1)
        noise_series = tf.unstack(n, axis=1)
        self.labels = y

        # mat_init = tf.zeros_initializer
        mat_init = tf.glorot_uniform_initializer
        # mat_init = tf.random_uniform_initializer
        self.Wxh = tf.get_variable("input_weights", [x.shape[-1], rnn_size], initializer=mat_init)
        self.Whh = tf.get_variable("hidden_weights", [rnn_size, rnn_size], initializer=mat_init)
        self.Wout = tf.get_variable("output_weights", [rnn_size, y.shape[-1]], initializer=mat_init)

        self.Whh_mask = tf.ones([1, rnn_size]) - tf.eye(rnn_size)
        # self.Wh_bias = tf.Variable(tf.zeros([1, rnn_size]), name="hidden_bias")
        # self.Wout_bias = tf.Variable(tf.zeros([1,y.shape[-1]]), name="output_bias")
        self.Wh_bias = tf.get_variable("hidden_bias", [1, rnn_size], initializer=tf.zeros_initializer)
        self.Wout_bias = tf.get_variable("output_bias", [1, y.shape[-1]], initializer=tf.zeros_initializer)

        next_state = tf.zeros(shape=[batch_size, rnn_size], dtype=tf.float32)  # init_state
        state_series = []
        logit_series = []

        print('building timesteps...')
        for i, (current_input, noise) in enumerate(zip(inputs_series, noise_series)):
            next_state, next_logit = self.scan_fn(next_state, current_input, noise, opts)
            state_series.append(next_state)
            logit_series.append(next_logit)

        print('building loss functions...')
        self.predictions = tf.stack([tf.nn.softmax(log) for log in logit_series], axis=1)
        xe = [tf.nn.softmax_cross_entropy_with_logits_v2(labels=lab, logits=log)
               for lab, log in zip(labels_series, logit_series)]

        self.states = tf.stack(state_series, axis=1)
        self.logits = tf.stack(logit_series, axis=1)
        self.error_loss = tf.reduce_mean(xe[time_loss_start:time_loss_end])
        # self.error_loss = tf.reduce_mean(xe)
        self.activity_loss = opts.activity_alpha * tf.reduce_mean(tf.square(self.states))
        self.weight_loss = opts.weight_alpha * tf.reduce_mean(tf.square(self.Whh))
        self.total_loss = self.error_loss + self.weight_loss + self.activity_loss

    def scan_fn(self, prev_state, input, noise, opts):
        Wxh, Whh, Wout = self.get_connections(opts)

        """
        x1 = (1.-dt/tau)*x1prev + (dt/tau)*tf.nn.relu(tf.matmul(x1prev,J1) + tf.matmul(x,wx) + b1 + sigeta_t*tf.random_normal([B,N]))
        x2 = (1.-dt/tau)*x2prev + (dt/tau)*tf.nn.relu(tf.matmul(x2prev,J2) + tf.matmul(x1,J12*J12mask) + b2 + sigeta_t*tf.random_normal([B,N]))
        x3 = (1.-dt/tau)*x3prev + (dt/tau)*tf.nn.relu(tf.matmul(x3prev,J3) + tf.matmul(x2,J23*J23mask) + b3 + sigeta_t*tf.random_normal([B,N]) + u3)
        y = tf.matmul(x3,wy)
        """

        decay = .2
        hidden_act = tf.matmul(prev_state, Whh) + tf.matmul(input, Wxh) + self.Wh_bias
        if opts.noise:
            hidden_act += noise
            # hidden_act += opts.noise_intensity * tf.random_normal([opts.batch_size, opts.rnn_size]))
        if opts.decay:  # decaying network
            # state = (1. - self.decay) * prev_state + self.decay * self.fn(hidden_act)
            # state = (1. - self.tau) * prev_state + self.tau * self.fn(hidden_act)
            state = (1 - decay) * prev_state + decay * self.fn(hidden_act)
        else:
            state = self.fn(hidden_act)

        logit = tf.matmul(state, Wout) + self.Wout_bias
        return [state, logit]

    def get_connections(self, opts):
        if opts.mask:
            Whh = self.Whh * self.Whh_mask
            # Whh = tf.multiply(self.Whh, self.Whh_mask)
        else:
            Whh = self.Whh

        Wxh = self.Wxh
        Wout = self.Wout

        # later mods for EI stuff
        return Wxh, Whh, Wout



