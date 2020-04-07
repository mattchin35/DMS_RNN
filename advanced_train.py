import torch
from numpy import random
import random
import time
import advanced_model
import os
import pickle as pkl

import numpy as np
from datasets import inputs
import config
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from utils.tools import torch2numpy
from utils.train_init import _initialize


def train(modelConfig, reload, set_seed=True, stop_crit=5.0):
    """Training program for use with XWJ networks"""
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
            hidden = (net.initialZeroState(), torch.zeros(opts.batch_size, opts.rnn_size))
            optimizer.zero_grad()
            loss_activity = 0
            loss_weight = 0
            loss_pred = 0

            hs, rs = [], []
            for t in range(x.shape[1]):
                xt = torch.Tensor(x[:,t,:])
                # yt = torch.Tensor(y[:,t,:])
                yt = torch.argmax(y[:,t,:], dim=1)
                hidden, out = net(xt, hidden)
                ht, rt = hidden
                ht.retain_grad()
                if t >= t_loss_start and t <= t_loss_end:
                    loss_activity += opts.activity_alpha * torch.mean(torch.pow(rt,2))
                    loss_weight += opts.weight_alpha * torch.mean(torch.pow(net.h_w,2))  # L2 weight loss
                    # loss_weight += opts.weight_alpha * torch.mean(torch.mean(net.h_w))  # L1 weight loss
                    loss_pred += criterion(out, yt)

                hs.append(ht)
                rs.append(rt)

            loss = loss_pred + loss_weight + loss_activity
            loss.backward()

            # Vanishing gradient regularization
            dxt = [h.grad.detach() for h in hs[1:]]
            _num = [(1 - net.time_const) * d +
                    net.time_const * torch.matmul(d, net.h_w) * (r > 0).float()
                    for d, r in zip(dxt, rs[:-1])]
            num = torch.sum(torch.stack(_num, dim=1) ** 2, dim=2)
            denom = torch.sum(torch.stack([d ** 2 for d in dxt], dim=1), dim=2)  # B x T
            omega = torch.mean((num / denom - 1) ** 2, dim=[0, 1])  # B x T
            vanishing_gradient_loss = omega * opts.vanishing_gradient_mult
            vanishing_gradient_loss.backward()

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
            print('Examples/second {:.1f}'.format(pe * opts.n_input / time_spent))

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
        hidden = (net.initialZeroState(), net.initialZeroState())

        xs, ys, youts, hs, rs = [], [], [], [], []
        for t in range(x.shape[1]):
            xt = torch.Tensor(x[:,t,:])
            yt = torch.Tensor(y[:,t,:])
            hidden, out = net(xt, hidden)

            ht, rt = hidden
            xs.append(torch2numpy(xt))
            ys.append(torch2numpy(yt))
            youts.append(torch2numpy(out))
            hs.append(torch2numpy(ht))
            rs.append(torch2numpy(rt))

        logger['x'] = np.array(xs)
        logger['y'] = np.array(ys)
        logger['y_out'] = np.array(youts)
        logger['h'] = np.array(hs)
        logger['r'] = np.array(rs)
        break

    for k, v in logger.items():
        logger[k] = np.stack(v, axis=1)

    if log:
        with open(os.path.join(opts.save_path, 'test_log.pkl'), 'wb') as f:
            pkl.dump(logger, f)
    return logger


if __name__ == "__main__":
    # c = config.XJWModelConfig()
    c = config.XJW_EIConfig()
    # c = config.load_config(c.save_path)
    c.clip_gradient = True
    c.vanishing_gradient_mult = 0
    c.trial_time['delay'] = .5
    c.epoch = 500
    train(c, reload=c.reload, set_seed=True)
    evaluate(c, log=True)
