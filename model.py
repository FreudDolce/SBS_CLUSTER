#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Ji Hongchen
# @Email : jhca.xyt@163.com
# @Last version : 2020-10-29 20:50
# @Filename : model.py

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cfg
from numpy import exp, power

CFG = cfg.CFG()


class LSTMGene(nn.Module):
    """
    For LSTM process in LSTM-SOM model.
    para: input_size: input_size
    para: hidden_size: LSTM hidden_size
    n_layers: LSTM n_layers
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 n_layers: int):
        super(LSTMGene, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstmgene = nn.LSTM(input_size,
                                hidden_size,
                                num_layers=n_layers,
                                bias=True,
                                batch_first=True,
                                bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=output_size),
            nn.BatchNorm1d(num_features=output_size),
        )

    def forward(self, x):                # x:[seq_size, batchsize, dim]
        # h_n/c_n:[n_layer, batch_size, hidden_size]
        lstm_output, (h_n, c_n) = self.lstmgene(x)
        output_last_time = h_n[-1, :, :]
        lstm_outlayer = self.fc(output_last_time)
        return lstm_outlayer


class SOMGene():
    """
    para :
        weight: inital_weight
        ouput_size: inital feature numbers
        target_lr
        sigma
    """

    def __init__(self, weight=None, target_lr=0.3, sigma=0.3):
        self.weight = weight
        self.target_lr = target_lr
        self.sigma = float(sigma)

    def Decay(self, distance):
        d = 2 * 3.14 * self.sigma * self.sigma
        decay = exp(-power(distance, 2) / d)
        return decay

    def CaluSom(self, batch_sample):
        batchsize = batch_sample.shape[0]
        batch_sample_add = np.zeros((batch_sample.shape))
        try:
            batch_weight_add = np.zeros((self.weight.shape))
        except AttributeError:
            self.weight = np.random.random(self.weight.shape)
            batch_weight_add = np.zeros((self.weight.shape))
        for i in range(batchsize):
            onesample_weigh_add = np.zeros(self.weight.shape)
            dist_pool = []
            for j in range(self.weight.shape[0]):
                w_t_dist = np.linalg.norm(batch_sample[i] - self.weight[j])
                dist_pool.append(w_t_dist)
            sort_pool = sorted(dist_pool)
            bun_loc = dist_pool.index(min(dist_pool))
            min_w_t_dist = min(dist_pool)
            push_w_t_dist = sort_pool[int(len(dist_pool) * CFG.PUSHTHRESHOLD)]
            for j in range(self.weight.shape[0]):
                dist_to_cent = np.linalg.norm(
                    self.weight[j] - self.weight[bun_loc])
                decay = self.Decay(dist_to_cent)
                dist_to_target = np.linalg.norm(
                    self.weight[j] - batch_sample[i])
                if dist_to_target <= push_w_t_dist:
                    abs_to_t_add = self.weight[j] - batch_sample[i]
                else:
                    abs_to_t_add = -1 * (self.weight[j] - batch_sample[i])
                to_t_weight_add = abs_to_t_add * self.target_lr * \
                    decay * min_w_t_dist / dist_to_target
                batch_sample_add[i] += to_t_weight_add
                onesample_weigh_add[j] += to_t_weight_add
            batch_weight_add += onesample_weigh_add
        self.weight -= batch_weight_add
        return batch_sample_add
