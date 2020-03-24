import tensorflow as tf
import config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from torch_model import Model


class MultilayerRNN(Model):
    def __init__(self, data, opts, training=True):
        super(MultilayerRNN, self).__init__(opts.save_path)

        X_pl, Y_pl = data[0], data[1]
        self.opts = opts
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(X_pl, Y_pl)

        print('built')

        if training:
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

    def _build(self, x, y):
        print('build start')
        opts = self.opts
        time_loss_start = opts.time_loss_start
        time_loss_end = opts.time_loss_end
        batch_size = opts.batch_size
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
            self.fn = lambda L: tf.nn.relu(tf.nn.tanh(L))

        inputs_series = tf.unstack(x, axis=1)
        labels_series = tf.unstack(y, axis=1)

        layer_size = opts.layer_size
        layer_size.insert(0, x.shape[-1])

        # EI_in = opts.EI_in  # either a percentage excitatory/inhibitory for each layer or None for random init
        # EI_h = opts.EI_h
        # EI_out = opts.EI_out

        self.Wxh, self.Whh, self.Wh_bias, self.Whh_mask, self.recurrent_mask, self.forward_mask, init_state = \
            [], [], [], [], [], [], []
        for i in range(1, len(layer_size)):
            prev = layer_size[i-1]
            cur = layer_size[i]
            self.Wxh.append(tf.get_variable(f"input_weights_{i-1}", [prev, cur]))
            self.Whh.append(tf.get_variable(f"hidden_weights_{i-1}", [cur, cur]))
            self.Wh_bias.append(tf.Variable(tf.zeros([1, cur]), name=f"hidden_bias_{i-1}"))
            self.Whh_mask.append(1-tf.eye(cur))

        self.Wout = tf.get_variable("output_weights", [layer_size[-1], y.shape[-1]])
        self.Wout_bias = tf.Variable(tf.zeros([1, y.shape[-1]]), name="output_bias")

        # layer_size.pop(0)
        next_state = [tf.zeros(shape=[batch_size, L], dtype=tf.float32) for L in layer_size[1:]]
        state_series = []
        logit_series = []

        for i, current_input in enumerate(inputs_series):
            next_state, next_logit = self.scan_fn(next_state, current_input, opts)
            state_series.append(next_state)
            logit_series.append(next_logit)

        self.predictions = [tf.nn.softmax(log) for log in logit_series]
        xe = [tf.nn.softmax_cross_entropy_with_logits_v2(labels=lab, logits=log)
               for lab, log in zip(labels_series, logit_series)]

        self.error_loss = tf.reduce_mean(xe[time_loss_start:time_loss_end])

        rnn_activity = tf.stack([tf.stack([s for s in state], axis=2) for state in state_series], axis=1)
        self.activity_loss = opts.activity_alpha * tf.reduce_mean(tf.square(rnn_activity))  # zero activity
        self.weight_loss = opts.weight_alpha * (tf.reduce_mean([tf.reduce_mean(tf.square(W)) for W in self.Whh]) +
                                                tf.reduce_mean([tf.reduce_mean(tf.square(W)) for W in self.Wxh]))
        self.total_loss = self.error_loss + self.weight_loss + self.activity_loss

        layer_ix = np.cumsum(layer_size)
        self.states = [rnn_activity[:, :, layer_size[i]:layer_size[i + 1]] for i in range(len(layer_ix) - 1)]
        self.logits = tf.stack(logit_series, axis=1)

    def scan_fn(self, prev_state, input, opts):
        # prev_state holds states for all previous layers
        new_state = []
        for i, prev in enumerate(prev_state):
            if opts.mask:
                Whh = self.Whh[i] * self.Whh_mask[i]
            else:
                Whh = self.Whh[i]

            Wxh = self.Wxh[i]
            hidden_act = tf.matmul(prev, Whh) + tf.matmul(input, Wxh) + self.Wh_bias[i]
            if opts.noise:
                hidden_act += opts.noise_intensity * tf.random_normal([opts.batch_size, opts.layers[i]])

            if opts.decay:  # decaying network
                state = (1. - self.decay) * prev + self.decay * self.fn(hidden_act)
            else:
                state = self.fn(hidden_act)

            new_state.append(state)
            input = state

        logit = tf.matmul(new_state[-1], self.Wout) + self.Wout_bias
        return [new_state, logit]

