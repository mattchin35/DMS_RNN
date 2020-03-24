import tensorflow as tf
import os
import pickle as pkl
import config


class Model(object):
    '''abstract model class'''
    def __init__(self, save_path):
        '''make model'''
        if save_path is None:
            save_path = os.getcwd()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.saver = None
        self.opts = None
        self.weight_dict = None

    def save(self, path=None):
        if path is not None:
            save_path = path
            # save_path = os.path.join(path, self.opts.losses)
        else:
            save_path = self.save_path
            # save_path = os.path.join(self.save_path, self.opts.losses)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = os.path.join(save_path, self.opts.model_name)
        sess = tf.get_default_session()
        save_path = self.saver.save(sess, save_path)
        print("[***] Model saved in path: %s" % save_path)

    def load(self):
        save_path = self.save_path
        # save_path = os.path.join(self.save_path, self.opts.losses)
        save_path = os.path.join(save_path, self.opts.model_name)
        sess = tf.get_default_session()
        self.saver.restore(sess, save_path)
        print("[***] Model restored from path: %s" % save_path)

    def save_weights(self, path=None):
        '''save model using pickle'''
        if path is not None:
            save_path = path
            # save_path = os.path.join(path, self.opts.losses)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = self.save_path
            # save_path = os.path.join(self.save_path, self.opts.losses)

        f_name = os.path.join(save_path, self.opts.weight_name)
        sess = tf.get_default_session()
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # vars = tf.all_variables()
        # vars = self.weight_dict
        var_dict = {v.name: sess.run(v) for v in vars}
        with open(f_name + ".pkl", 'wb') as f:
            pkl.dump(var_dict, f)
        print("[***] Model weights saved in path: %s" % save_path)

