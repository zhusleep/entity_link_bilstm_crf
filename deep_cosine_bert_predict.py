#-*-coding:utf-8-*-
import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager,read_kb
from spo_dataset import Entity_Vector, get_mask, collate_fn_link_entity_vector
from spo_model import SPOModel, EntityLink_entity_vector
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1, get_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
import logging
import time
from torch.nn import functional as F
from sklearn.model_selection import KFold, train_test_split
from sklearn.externals import joblib

file_namne = 'data/raw_data/train.json'
train_part = data_manager.read_deep_cosine_test(None)
seed_torch(2019)
print('train size %d ', len(train_part))


data_all = np.array(train_part)
kfold = KFold(n_splits=5, shuffle=False, random_state=2019)
pred_vector = []
round = 0
for r in range(2):

    valid_part = data_all

    BERT_MODEL = 'bert-base-chinese'
    CASED = False
    t = BertTokenizer.from_pretrained(
        BERT_MODEL,
        do_lower_case=True,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        #    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
    )

    valid_dataset = Entity_Vector([['[CLS]'] + list(x[0]) + ['[CLS]'] for x in valid_part], t,
                                  pos=[[x[1][0] + 1, x[1][1] + 1] for x in valid_part],
                                  vector=[x[2] for x in valid_part],
                                  label=[x[3] for x in valid_part],
                                  use_bert=True)
    valid_batch_size = 64

    model = EntityLink_entity_vector(vocab_size=None, init_embedding=None,
                       encoder_size=128, dropout=0.2, num_outputs=len(data_manager.type_list),
                                     use_bert=True)
    model.load_state_dict(torch.load('entity_embedding/gensim_vector_bert_model_round %s' % (round)))

    use_cuda=True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)

    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_link_entity_vector, shuffle=False, batch_size=valid_batch_size)


    model.eval()
    valid_loss = 0
    valid_cosloss = 0
    pred_set = []
    label_set = []
    for index, X, label, pos, vector, length in valid_dataloader:
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.to(device)
        length = length.to(device)

        # ner = ner.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=use_cuda).to(device)
        pos = pos.type(torch.LongTensor).to(device)
        label = label.to(device).type(torch.float)
        vector = vector.to(device).type(torch.float)

        with torch.no_grad():
            pred = model(X, mask_X, pos, vector, length)
        # cos_loss = torch.nn.MSELoss(pred, vector).item()
        # print('loss',loss)
        # valid_cosloss +=cos_loss
        pred_set.append(pred.cpu().numpy())
        # label_set.append(type.cpu().numpy())

    pred_set = np.concatenate(pred_set, axis=0)

    pred_vector.append(pred_set)
    round += 1
    # INFO_THRE, thre_list = get_threshold(pred_set, label_set, len(data_manager.type_list))
    # INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    # logging.info(INFO + '\t' + INFO_THRE)
pred_vector = np.concatenate(pred_vector, axis=1)
np.save('entity_embedding/gensim_vector_bert_predict.npy', pred_vector)
    # 19 train loss　0.128424, val loss 0.153328, val_cos_loss 91155.319716



