#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Ji Hongchen
# @Email : jhca.xyt@163.com
# @Last version : 2020-10-28 21:14
# @Filename : dataloader.py

import numpy as np
import pandas as pd
from pyfaidx import Fasta
import cfg
import os

CFG = cfg.CFG()


def GetWholeSequence(seq_path):
    """
    Return the whole sequences for further use
    """
    wholeseq = Fasta(seq_path)
    return wholeseq
WHOLE_SEQ = GetWholeSequence(CFG.WHOLE_SEQ_PATH)


def _get_mut_dict():
    """
    return the code dict of genes
    """
    mutdict = {
        'A': [0.0, 0.0],
        'T': [0.0, 1.0],
        'C': [1.0, 0.0],
        'G': [1.0, 1.0],
        'N': [-1.0, -1.0]
    }
    return mutdict


def _get_patient_id(name):
    """
    Return patient id according to Tumor_Sample_Barcode
    """
    p_name = name.split('-')
    p_id = '-'.join(p_name[0:3])
    return p_id


def _read_maf_data(maf_file):
    kwargs = {'sep': '\t', 'comment': '#'}
    mut_info = pd.read_csv(maf_file, **kwargs)
    mut_list = []
    for i in range(len(mut_info)):
        mut_list.append([_get_patient_id(mut_info['Tumor_Sample_Barcode'][i]),
                         mut_info['Hugo_Symbol'][i],
                         mut_info['Chromosome'][i],
                         mut_info['Start_Position'][i],
                         mut_info['End_Position'][i],
                         mut_info['Reference_Allele'][i],
                         mut_info['Tumor_Seq_Allele1'][i],
                         mut_info['Tumor_Seq_Allele2'][i]])
    mut_list = np.array(mut_list)
    return mut_list


def GetMutSeq(geneseq, mutpos, extend=30, bidirct=False):
    """
    parameters:geneseq, mutpos
    geneseq:genes
    mutpos:[chr, start_position, ref_all, t_all_1, t_all_2]
    return (pre_list(extend before mut), post_list(extend after mut))
    """
    mutdict = _get_mut_dict()
    pre_list = str(geneseq.get_seq(
        mutpos[0], mutpos[1]-extend, mutpos[1]-1)).upper()
    post_list = str(geneseq.get_seq(
        mutpos[0], mutpos[1]+1, mutpos[1]+extend)).upper()
    post_list = post_list[:: -1]
    mut_mat = np.array([mutdict[mutpos[3][0]],
                        mutdict[mutpos[4][0]]]).reshape(1, -1)
    for base in pre_list:
        gene_mat = np.array(mutdict[base]).reshape(1, -1)
        gene_mat = np.hstack((gene_mat, gene_mat))
        try:
            pre_list_mat = np.vstack((pre_list_mat, gene_mat))
        except UnboundLocalError:
            pre_list_mat = gene_mat
    pre_list_mat = np.r_[pre_list_mat, mut_mat]
    for post_base in post_list:
        gene_mat = np.array(mutdict[post_base]).reshape(1, -1)
        gene_mat = np.hstack((gene_mat, gene_mat))
        try:
            post_list_mat = np.vstack((post_list_mat, gene_mat))
        except UnboundLocalError:
            post_list_mat = gene_mat
    post_list_mat = np.r_[post_list_mat, mut_mat]
    if bidirct == True:
        output_list_mat = np.hstack((pre_list_mat, post_list_mat))
        return output_list_mat
    else:
        return (pre_list_mat, post_list_mat)


def GetBatchSample(file_name, grade, batchsize=CFG.BATCH_SIZE):
    """
    return items/batchsize of train batches,
    as size of: (n_bch, bch_size, lenth, dim)
    return format: train_batches, left_samples
    !!!!! -> if sigle bidirct is needed, modify item_sample into _, item_sample
                                       and bidirct=False
    """
    mutationdf = pd.read_csv(file_name)
    if 'class' in mutationdf.columns:
        mutationdf = mutationdf[mutationdf['class'] == int(grade[1:])]
    mutation_array = np.array(mutationdf)
    total_items = len(mutation_array)
    arr = np.arange(total_items)
    np.random.shuffle(arr)
    batch_num = int(total_items//batchsize) + 1
    print('====================================================================')
    print('File ' + file_name + ' is loading...')
    print('Total batch = ' + str(batch_num))
    train_batches = None
    for num in range(batch_num):
        batch_sample = None
        try:
            for j in range(num * batchsize, (num + 1) *
                           batchsize):
                item_sample = GetMutSeq(geneseq=WHOLE_SEQ,
                                                   mutpos=[mutation_array[j][2],
                                                           mutation_array[j][3],
                                                           mutation_array[j][5],
                                                           mutation_array[j][6],
                                                           mutation_array[j][7]],
                                                   extend=CFG.EXTEND_NUM,
                                                   bidirct=True)
                item_sample = item_sample.reshape(1, CFG.EXTEND_NUM+1,
                                                  CFG.LSTMINPUTSIZE)
                try:
                    batch_sample = np.vstack((batch_sample, item_sample))
                except NameError:
                    batch_sample = item_sample
                except ValueError:
                    batch_sample = item_sample
        except IndexError:
            try:
                left_samples = batch_sample.reshape(1, -1,
                                                   CFG.EXTEND_NUM+1,
                                                   CFG.LSTMINPUTSIZE)
            except AttributeError:
                left_samples = 0
            continue
        try:
            train_batches = np.vstack((train_batches, batch_sample.reshape(1, -1,
                                                                       CFG.EXTEND_NUM+1,
                                                                       CFG.LSTMINPUTSIZE)))
        except NameError:
            train_batches = batch_sample.reshape(1, -1,
                                               CFG.EXTEND_NUM+1,
                                               CFG.LSTMINPUTSIZE)
        except ValueError:
            train_batches = batch_sample.reshape(1, -1,
                                               CFG.EXTEND_NUM+1,
                                               CFG.LSTMINPUTSIZE)
    print(file_name + ' loaded.')
    print('Get sample in file: ', train_batches.shape, '.')
    print('====================================================================')
    return train_batches, left_samples         # left_sample:samples cannot be devided


if __name__ == '__main__': #devide .maf file into .csv file randomly
    #samples, left_samples = GetBatchSample('/home/ji/Documents/lstmsom_data/mut_info_20200212/mut_info_130.csv')
    #print(samples.shape, left_samples.shape)
    folders = os.listdir('/media/ji/data/TCGA_DATA/MAF/')
    outputtlist = None
    for folder in folders:
        files = os.listdir('/media/ji/data/TCGA_DATA/MAF/' + folder)
        for file in files:
            if ('somaticsniper' in file):
                mutlist = ReadMafData(
                    '/media/ji/data/TCGA_DATA/MAF/' + folder + '/' + file)
                print(file + 'listed.')
                print(
                    '===========================================================================')
                if outputtlist is None:
                    outputtlist = mutlist
                else:
                    outputtlist = np.vstack((outputtlist, mutlist))
                print(outputtlist.shape)
    len_outputlist = len(outputtlist)
    col_name = ['Tumor_Sample_Barcode', 'Hugo_Symbol', 'Chromosome', 'Start_Position',
                'End_Position', 'Reference_Allele', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2']
    for i in range(len_outputlist):
        print(outputtlist[i])
        lucky_num = np.random.choice(len_outputlist)
        file_num = int(lucky_num // 5000)  # 5000 items per file
        print('lucky_num=' + str(file_num))
        try:
            existed_list = pd.read_csv(
                TARGET_FOLDER + 'mut_info_' + str(file_num) + '.csv')
            existed_list = np.array(existed_list)
            output_list = np.vstack((existed_list, outputtlist[i]))
            output_list = pd.DataFrame(output_list)
            output_list.to_csv(TARGET_FOLDER + 'mut_info_' + str(file_num) + '.csv',
                               header=col_name,
                               index=None)
        except FileNotFoundError:
            output_list = pd.DataFrame([outputtlist[i]])
            output_list.to_csv(TARGET_FOLDER + 'mut_info_' + str(file_num) + '.csv',
                               header=col_name,
                               index=None)
