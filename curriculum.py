import torch_train
import torch_model
import advanced_train
import advanced_model
import config


def curriculum():
    # c = config.oneLayerModelConfig()
    c = config.EIModelConfig()
    c.trial_time['delay'] = .5
    c.epoch = 500
    c.clip_gradient = True
    # c = torch_model.load_config(c.save_path)
    torch_train.train(c, reload=c.reload, set_seed=True)

    c = torch_model.load_config(c.save_path, c.mode)
    c.trial_time['delay'] = 1.5
    c.epoch = 200
    c.reload = True
    torch_train.train(c, reload=c.reload, set_seed=True)

    torch_train.evaluate(c, log=True)


def advanced_curriculum():
    # c = config.XJWModelConfig()
    c = config.XJW_EIConfig()
    c.trial_time['delay'] = .5
    c.epoch = 500
    c.clip_gradient = True
    # c = torch_model.load_config(c.save_path)
    advanced_train.train(c, reload=c.reload, set_seed=True)

    c = torch_model.load_config(c.save_path, c.mode)
    c.trial_time['delay'] = 1.5
    c.epoch = 200
    c.reload = True
    advanced_train.train(c, reload=c.reload, set_seed=True)

    advanced_train.evaluate(c, log=True)


if __name__ == "__main__":
    curriculum()