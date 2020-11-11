#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Ji Hongchen
# @Email : jhca.xyt@163.com
# @Last version : 2020-10-29 22:33
# @Filename : train.py


import numpy as np
import pandas as pd
import torch
from model import LSTMGene
from model import SOMGene
import cfg
from torch.autograd import Variable
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
from dataloader import GetBatchSample
import argparse
import json
matplotlib.use('Agg')

LOG_DICT = {'lstm_model': '', 'som_weight': '',
            'files_list': [], 'file_num': 0}

CFG = cfg.CFG()

parser = argparse.ArgumentParser()
parser.add_argument('-g', help="Grade of training.")
args = parser.parse_args()

GRADE = args.g

GENE_LSTM = LSTMGene(input_size=CFG.LSTMINPUTSIZE,
                     hidden_size=CFG.LSTMHIDSIZE,
                     output_size=CFG.LSTMOUTPUTSIZE,
                     n_layers=CFG.LSTM_NUM_LAYERS)

GENE_LSTM = GENE_LSTM.cuda()
LSTM_OPTIMIZER = torch.optim.Adam(GENE_LSTM.parameters(), lr=CFG.LSTMLR)

WEIGHTS = np.random.rand(CFG.WEIGHTSHAPE[0], CFG.WEIGHTSHAPE[1]) - 0.5
files_list = os.listdir(CFG.MUT_INFO_PATH)

SOM_GENE = SOMGene(weight=WEIGHTS,
                   target_lr=CFG.SOMTARLR,
                   sigma=CFG.SIGMA
                   )

LSTM_LOSSFUNC = nn.MSELoss()
LSTM_LOSSFUNC = LSTM_LOSSFUNC.cuda()

fig = plt.figure()

for num in range(len(files_list)):
    samples, left_sample = GetBatchSample(
        CFG.MUT_INFO_PATH + files_list[num], grade=GRADE)
    for k in range(len(samples)):
        plt.cla()
        print('------------------------------------------------------------------------')
        print('Batch ', k + 1, ' training >>>>')
        batch_v = torch.from_numpy(samples[k])
        batch_var = torch.tensor(batch_v, dtype=torch.float32)
        batch_var = batch_var.cuda()
        for i in range(CFG.BATCH_ITER):
            lstm_out = GENE_LSTM(batch_var)
            lstm_out_np = lstm_out.cpu().detach().numpy()
            add_value = SOM_GENE.CaluSom(lstm_out_np)
            lstm_out_np = lstm_out_np + add_value
            _y = torch.from_numpy(lstm_out_np)
            y = torch.tensor(_y, dtype=torch.float32)
            y = y.cuda()
            for j in range(CFG.LSTMITER):
                loss = LSTM_LOSSFUNC(lstm_out, y)
                LSTM_OPTIMIZER.zero_grad()
                loss.backward(retain_graph=True)
                LSTM_OPTIMIZER.step()
            if i % CFG.LOSS_PRINT == CFG.LOSS_PRINT - 1:
                print('File: ', files_list[num], ', Batch: ', k + 1, ', iter: ',
                      i + 1, '. | ', 'Loss: ', float(loss.data), '.')
        ax = Axes3D(fig)
        ax.scatter(lstm_out_np[:, 0],
                   lstm_out_np[:, 1],
                   lstm_out_np[:, 2],
                   c='steelblue',
                   s=20,
                   alpha=0.50)
        figpath = CFG.FIG_PATH + '/' + str(GRADE)[0] + '/'
        if os.path.exists(figpath) == False:
            os.mkdir(figpath)
        plt.savefig(figpath+'file'+str(num+1)+'_batch'+str(k+1)+'.jpg')
        if (k == int(len(samples)/CFG.EXP_SAVE_TIME)-1) or (k == int(len(samples))-1):
            model_fname = 'file'+str(num+1)+'_'+GRADE+'_batch'+str(k+1)
            print(
                '******************************************************************************************')
            print(model_fname, ' is now saving...')
            torch.save({'model_state_dict': GENE_LSTM.state_dict(),
                        'optim_state_dict': LSTM_OPTIMIZER.state_dict()},
                       CFG.MODEL_PATH+model_fname+'.tar')
            print('---> lstm model and optimizer were saved.')
            LOG_DICT['lstm_model'] = CFG.MODEL_PATH+model_fname+'.tar'
            np.save(file=CFG.MODEL_PATH+model_fname +
                    '.npy', arr=SOM_GENE.weight)
            print('---> som weight saved.')
            LOG_DICT['som_weight'] = CFG.MODEL_PATH+model_fname+'.npy'
            LOG_DICT['files_list'] = files_list
            LOG_DICT['file_num'] = num
            print('---> file number saved.')
            log_json = json.dumps(LOG_DICT)
            jsonfile = open(CFG.LOG_PATH+model_fname+'.json', 'w')
            jsonfile.write(log_json)
            jsonfile.close()
            print(
                '******************************************************************************************')
    print('******************************************************************************************')
    print('---> file list refreshed.')
    log_json = json.dumps(LOG_DICT)
    jsonfile = open(CFG.LOG_PATH+model_fname+'.json', 'w')
    jsonfile.write(log_json)
    jsonfile.close()
    print('******************************************************************************************')
