import torch
from numpy import random
import random
import time
import torch_model
import os
import pickle as pkl
import config

import numpy as np
from datasets import inputs
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from utils.tools import torch2numpy
from utils.train_init import _initialize


def train(modelConfig, reload, set_seed=True, stop_crit=5.0):
    opts, data_loader, net = _initialize(modelConfig, reload, set_seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0 * opts.learning_rate)

    n_epoch = opts.epoch
    logger = defaultdict(list)
    t_loss_start = opts.time_loss_start
    t_loss_end = opts.time_loss_end
    print("Starting training...")

    start_time = time.time()
    total_time = 0
    if not reload:
        net.save('net', 0)
    for ep in range(n_epoch):
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.CrossEntropyLoss()
        for x, y in data_loader:
            hidden = net.initialZeroState()
            optimizer.zero_grad()
            loss_activity = 0
            loss_weight = 0
            loss_pred = 0

            for t in range(x.shape[1]):
                xt = torch.Tensor(x[:,t,:])
                # yt = torch.Tensor(y[:,t,:])
                yt = torch.argmax(y[:,t,:], dim=1)
                hidden, out = net(xt, hidden)
                if t >= t_loss_start and t <= t_loss_end:
                    if opts.mode == 'three_layer':
                        h0, h1, h2 = hidden
                        act_mean = torch.mean(torch.pow(h0, 2)) + torch.mean(torch.pow(h1, 2)) + torch.mean(
                            torch.pow(h2, 2))
                        weight_mean = torch.mean(torch.pow(net.i_to_h0, 2)) + torch.mean(torch.pow(net.h0_w, 2)) \
                                      + torch.mean(torch.pow(net.h1_w, 2)) + torch.mean(torch.pow(net.h0_to_h1, 2)) \
                                      + torch.mean(torch.pow(net.h2_w, 2)) + torch.mean(torch.pow(net.h1_to_h2, 2))
                        loss_weight += opts.weight_alpha * weight_mean
                    else:
                        act_mean = torch.mean(torch.pow(hidden, 2))
                        weight_mean = torch.mean(torch.pow(net.h_w,2))

                    loss_activity += opts.activity_alpha * act_mean
                    loss_weight += opts.weight_alpha * weight_mean
                    loss_pred += criterion(out, yt)

            loss = loss_pred + loss_weight + loss_activity
            loss.backward()

            if opts.clip_gradient:
                for n, p in net.named_parameters():
                    if p.requires_grad and torch.norm(p.grad) > 1:
                        p.grad *= 1 / torch.norm(p.grad)

            optimizer.step()

            logger['epoch'].append(ep)
            logger['loss'].append(torch2numpy(loss))
            logger['error_loss'].append(torch2numpy(loss_pred))
            logger['activity_loss'].append(torch2numpy(loss_activity))
            logger['weight_loss'].append(torch2numpy(loss_weight))


        pe = opts.print_epoch_interval
        se = opts.save_epoch_interval
        n_iter = opts.n_input // opts.batch_size
        cnt = ep+1
        if cnt % pe == 0 and ep != 0:
            print('[' + '*' * 50 + ']')
            print('Epoch {:d}'.format(cnt))
            print("Mean loss: {:0.2f}".format(np.mean(logger['loss'][-n_iter:])))
            print("Error loss: {0:.2f}, Weight loss: {1:.2f}, Activity loss: {2:.2f}".format(
                np.mean(logger['error_loss'][-n_iter:]),
                np.mean(logger['weight_loss'][-n_iter:]),
                np.mean(logger['activity_loss'][-n_iter:])))

            time_spent = time.time() - start_time
            total_time += time_spent
            start_time = time.time()
            print('Time taken {:0.1f}s'.format(total_time))
            print('Examples/second {:.1f}'.format(pe / time_spent))

        if np.mean(logger['error_loss'][-n_iter:]) < stop_crit:
            print("Training criterion reached. Saving files...")
            net.save('net', cnt)
            net.save('net')
            break

        if cnt % se == 0 and ep != 0:
            print("Saving files...")
            net.save('net', cnt)
            net.save('net')


def evaluate(modelConfig, log):
    print("Starting testing...")

    opts, data_loader, net = _initialize(modelConfig, reload=True, set_seed=False, test=True)
    logger = defaultdict(list)

    for x, y in data_loader:
        hidden = net.initialZeroState()

        xs, ys, youts, hs = [], [], [], []
        for t in range(x.shape[1]):
            xt = torch.Tensor(x[:,t,:])
            yt = torch.Tensor(y[:,t,:])
            hidden, out = net(xt, hidden)
            xs.append(torch2numpy(xt))
            ys.append(torch2numpy(yt))
            youts.append(torch2numpy(out))
            hs.append(torch2numpy(hidden))

        logger['x'] = np.array(xs)
        logger['y'] = np.array(ys)
        logger['y_out'] = np.array(youts)
        logger['h'] = np.array(hs)
        break

    for k, v in logger.items():
        logger[k] = np.stack(v, axis=1)

    if log:
        #batch, time, neuron
        with open(os.path.join(opts.save_path, 'test_log.pkl'), 'wb') as f:
            pkl.dump(logger, f)
    return logger

if __name__ == "__main__":
    # c = config.oneLayerModelConfig()
    # c = config.EIModelConfig()
    c = config.threeLayerModelConfig()
    # c.trial_time['delay'] = .5
    c.clip_gradient = True
    c.epoch = 500
    # c = config.load_config(c.save_path)
    train(c, reload=c.reload, set_seed=True)
    # evaluate(c, log=True)
