#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Ji Hongchen
# @Email : jhca.xyt@163.com
# @Last version : 2020-10-30 16:32
# @Filename : test.py

import numpy as np
import pandas as pd
import torch
from model import LSTMGene
from model import SOMGene
import cfg
from torch.autograd import Variable
import torch.nn as nn
import os
import dataloader
import json
import argparse
from datetime import datetime
import time


CFG = cfg.CFG()
MODEL = '/home/ji/Documents/test_data/exp20200816/lstmsom_model/file10_22000_batch38.tar'
FILE_LIST = np.load('/home/ji/Documents/lstmsom_data/clinical_data/partmutfilelist.npy')
LIST_FROM = CFG.MUT_INFO_PATH
CLASS_COL = 'pred_4'
CLASS_LINE = [5, 0]
RETRAIN = False
# !!!!  Check the above para  !!!!

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=int, help='whether class or not, use 1 or 0')
parser.add_argument('-g', help='grade of training of testing')
args = parser.parse_args()

GRADE = args.g

WHOLE_SEQ = dataloader.GetWholeSequence(CFG.WHOLE_SEQ_PATH)
GENE_LSTM = LSTMGene(input_size=CFG.LSTMINPUTSIZE,
                     hidden_size=CFG.LSTMHIDSIZE,
                     output_size=CFG.LSTMOUTPUTSIZE,
                     n_layers=CFG.LSTM_NUM_LAYERS)
checkpoint = torch.load(MODEL)
GENE_LSTM.load_state_dict(checkpoint['model_state_dict'])
GENE_LSTM.eval()
print(MODEL + ' is loaded.')

def GetClinclInfo(path):
    clinic_info = pd.read_csv(path)
    return (clinic_info)



def GetTestResult(mut_item):
    pre_list_mat = dataloader.GetMutSeq(geneseq=WHOLE_SEQ,
                                        mutpos=[mut_item[2],
                                                mut_item[3],
                                                mut_item[5],
                                                mut_item[6],
                                                mut_item[7]],
                                        extend=CFG.EXTEND_NUM,
                                        bidirct=True)

    pre_list_mat = pre_list_mat.reshape(1, CFG.EXTEND_NUM+1,
                                        CFG.LSTMINPUTSIZE)
    pre_list_v = torch.from_numpy(pre_list_mat)
    pre_list_var = torch.tensor(pre_list_v, dtype=torch.float32)
    item_pred = GENE_LSTM(pre_list_var)
    item_pred_np = item_pred.detach().numpy()
    return (item_pred_np)


if __name__ == '__main__':
    i = 0
    save_file = 0
    wrong_num = 0
    file_list = os.listdir(LIST_FROM)
    for nu in range(len(FILE_LIST)):
        start_t = time.time()
        print('========================================================')
        start_t = time.time()
        print(FILE_LIST[nu], ' No.', nu+1, ' is now loading.')
        classedlist = pd.read_csv(LIST_FROM + '/' + FILE_LIST[nu])
        for c_n in CFG.PRED_INDEX:
            classedlist[c_n] = 0
        if 'class' not in classedlist:
            classedlist['class'] = 0
        if RETRAIN == True:
            classedlist['class'] = 0
        comb_result = []
        items = np.array(classedlist)
        for item in items:
            result = GetTestResult(item)
            if comb_result == []:
                comb_result = result
            else:
                comb_result = np.vstack((comb_result, result))
        classedlist[CFG.PRED_INDEX] = comb_result
        if args.c != 0:
            classedlist['new_class'] = classedlist['class']
            for ind in range(len(CLASS_LINE)):
                classedlist['new_class'][(classedlist['class'] == int(GRADE[1:])) & (classedlist[CLASS_COL] <= CLASS_LINE[ind])] += \
                        10 ** (4-int(GRADE[0]))
            classedlist['class'] = classedlist['new_class']
            del classedlist['new_class']
            print (classedlist)
        else:
            print (classedlist[classedlist['class'] == int(GRADE[1:])])

        classedlist.to_csv(LIST_FROM + '/' + FILE_LIST[nu], index=False)
