#-*-coding:utf-8-*-
import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager,read_kb
from spo_dataset import SPO_LINK, get_mask, collate_fn_link
from spo_model import SPOModel, EntityLink
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1,get_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
import logging
import time


file_namne = 'data/raw_data/train.json'
train_X, train_pos, train_type, dev_X, dev_pos, dev_type = data_manager.read_entity_embedding(file_name=file_namne, valid_num=10000)
seed_torch(2019)

t = Tokenizer(max_feature=10000, segment=False, lowercase=True)
t.fit(train_X+dev_X)

print('一共有%d 个字' % t.num_words)

train_dataset = SPO_LINK(train_X, t, pos=train_pos, type=train_type)
valid_dataset = SPO_LINK(dev_X, t, pos=dev_pos, type=dev_type)

batch_size = 1


# 准备embedding数据
embedding_file = 'embedding/miniembedding_baike_link.npy'
#embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'

if os.path.exists(embedding_file):
    embedding_matrix = np.load(embedding_file)
else:
    embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
    #embedding = '/home/zhu/Desktop/word_embedding/sgns.baidubaike.bigram-char'
    #embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
    embedding_matrix = load_glove(embedding, t.num_words+100, t)
    np.save(embedding_file, embedding_matrix)
