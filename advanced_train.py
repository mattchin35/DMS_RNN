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
from tools import torch2numpy


class InputDataset(Dataset):
    def __init__(self, opts):
        X, Y = inputs.create_inputs(opts)
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


def _initialize(opts, reload, set_seed, test=False):
    np.set_printoptions(precision=2)
    if set_seed:
        seed = opts.rng_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    if test:
        assert opts.test_batch_size <= opts.n_input
        opts.batch_size = opts.test_batch_size

    use_cuda = torch.cuda.is_available()
    if opts.ttype == 'float':
        ttype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    else:
        ttype = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
    torch.set_default_tensor_type(ttype)

    dataset = InputDataset(opts)
    data_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)
    if opts.mode == 'XJW_simple':
        net = advanced_model.XJW_Simple(opts=opts, isize=dataset.X.shape[-1], osize=dataset.Y.shape[-1])
    elif opts.mode == 'XJW_EI':
        net = advanced_model.XJW_EI(opts=opts, isize=dataset.X.shape[-1], osize=dataset.Y.shape[-1])

    net.model_config = opts

    if reload:
        net.load(name='net')
    print('[***Saving Variables***]')
    for name, param in net.named_parameters():
        if param.requires_grad:
            print('{0:20}: {1}'.format(name, param.data.shape))

    opts.time_loss_end = int(np.sum([v for v in opts.trial_time.values()]) / opts.dt)
    if opts.fixation:
        opts.time_loss_start = 5
    else:
        opts.time_loss_start = opts.time_loss_end - int(opts.trial_time['response'] / opts.dt)
    return opts, data_loader, net


def advanced_train(modelConfig, reload, set_seed=True, stop_crit=5.0):
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
            hidden = net.initialZeroState()
            optimizer.zero_grad()
            loss_activity = 0
            loss_weight = 0
            loss_pred = 0

            hs, rs = [], []
            for t in range(x.shape[1]):
                xt = torch.Tensor(x[:,t,:])
                xt.retain_grad()
                # yt = torch.Tensor(y[:,t,:])
                yt = torch.argmax(y[:,t,:], dim=1)
                hidden, (rate, out) = net(xt, hidden)
                if t >= t_loss_start and t <= t_loss_end:
                    loss_activity += opts.activity_alpha * torch.mean(torch.pow(hidden,2))
                    loss_weight += opts.weight_alpha * torch.mean(torch.pow(net.h_w,2))  # L2 weight loss
                    # loss_weight += opts.weight_alpha * torch.mean(torch.mean(net.h_w))  # L1 weight loss
                    loss_pred += criterion(out, yt)

                hs.append(hidden)
                rs.append(rate)

                # Vanishing gradient regularization
            dxt = [h.grad for h in hs[1:]]
            _num = [(1 - opts.time_const) * d +
                    opts.time_const * torch.matmul(d, net.h_w) * (h > 0).float() for d, h in zip(dxt, rs[:-1])]
            num = torch.sum(torch.stack(_num, dim=1) ** 2, dim=2)
            denom = torch.sum(torch.stack([d ** 2 for d in dxt], dim=1), dim=2)  # B x T
            omega = torch.mean((num / denom - 1) ** 2, dim=[0, 1])  # B x T

            # loss = loss_pred + loss_weight + loss_activity
            loss = (loss_pred + loss_weight + loss_activity + omega * opts.vanishing_gradient_mult) / opts.batch_size
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
        with open(os.path.join(opts.save_path, 'test_log.pkl'), 'wb') as f:
            pkl.dump(logger, f)
    return logger

if __name__ == "__main__":
    c = config.XJWModelConfig()
    # c = torch_model.load_config(c.save_path)
    c.learning_rate = .01
    c.clip_gradient = True
    advanced_train(c, reload=c.reload, set_seed=True)
    evaluate(c, log=True)
