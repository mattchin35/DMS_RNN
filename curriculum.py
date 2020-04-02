import torch_train
import torch_model
import config


def curriculum():
    # c = config.oneLayerModelConfig()
    c = config.EIModelConfig()
    c.save_path = './_DATA/EI'
    c.trial_time['delay'] = .5
    c.rnn_size = 200
    c.epoch = 500
    # c = torch_model.load_config(c.save_path)
    torch_train.train(c, reload=c.reload, set_seed=True)

    c = torch_model.load_config(c.save_path, c.mode)
    c.trial_time['delay'] = 1.5
    c.epoch = 200
    c.reload = True
    torch_train.train(c, reload=c.reload, set_seed=True)

    torch_train.evaluate(c, log=True)


if __name__ == "__main__":
    curriculum()