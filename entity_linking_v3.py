#-*-coding:utf-8-*-
import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager,read_kb
from spo_dataset import entity_linking_v3, get_mask, collate_fn_linking_v2
from spo_model import SPOModel, EntityLink_v3
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1,get_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
import logging
import time


file_namne = 'data/raw_data/train.json'
train_part, valid_part = data_manager.parse_v2(file_name=file_namne, valid_num=10000)
seed_torch(2019)

t = Tokenizer(max_feature=10000, segment=False, lowercase=True)

train_dataset = entity_linking_v3(train_part, t)
valid_dataset = entity_linking_v3(valid_part, t)

batch_size = 1


# 准备embedding数据
embedding_file = 'embedding/miniembedding_baike_link.npy'
#embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'

if os.path.exists(embedding_file):
    embedding_matrix = np.load(embedding_file)
else:
    #embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
    embedding = '/home/zhu/Desktop/word_embedding/sgns.baidubaike.bigram-char'
    #embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
    embedding_matrix = load_glove(embedding, t.num_words+100, t)
    np.save(embedding_file, embedding_matrix)



train_batch_size = 64
valid_batch_size = 16

model = EntityLink_v3(vocab_size=embedding_matrix.shape[0], encoder_size=128,
                      dropout=0.2,
                      init_embedding=embedding_matrix)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_linking_v2, shuffle=True, batch_size=train_batch_size)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_linking_v2, shuffle=True, batch_size=valid_batch_size)

epoch = 20

loss_fn = nn.BCELoss()
use_cuda=True
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

optimizer = torch.optim.Adam(model.parameters())

model.to(device)
for epoch in range(epoch):
    model.train()
    train_loss = 0
    for index, label, query, l_query, pos, candidate_abstract, l_abstract,\
        candidate_labels, l_labels, candidate_type, candidate_abstract_numwords,\
            candidate_numattrs in tqdm(train_dataloader):
        model.zero_grad()
        query = nn.utils.rnn.pad_sequence(query, batch_first=True).type(torch.LongTensor).to(device)
        l_query = l_query.to(device)
        mask_query = get_mask(query, l_query, is_cuda=use_cuda).to(device).type(torch.float)

        candidate_abstract = nn.utils.rnn.pad_sequence(candidate_abstract, batch_first=True).type(torch.LongTensor).to(device)
        l_abstract = l_abstract.to(device)
        mask_abstract = get_mask(candidate_abstract, l_abstract, is_cuda=use_cuda).to(device).type(torch.float)

        candidate_labels = nn.utils.rnn.pad_sequence(candidate_labels, batch_first=True).type(torch.LongTensor).to(device)
        l_labels = l_labels.to(device)
        mask_labels = get_mask(candidate_labels, l_labels, is_cuda=use_cuda).to(device).type(torch.float)

        pos = pos.type(torch.LongTensor).to(device)

        candidate_type = candidate_type.to(device)
        label = label.type(torch.float).to(device).unsqueeze(1)
        candidate_numattrs = candidate_numattrs.to(device).type(torch.float)
        candidate_abstract_numwords = candidate_abstract_numwords.to(device).type(torch.float)
        #ner = ner.type(torch.float).cuda()
        #print(index)
        pred = model(query, l_query, pos, candidate_abstract, l_abstract,
                     candidate_labels, l_labels, candidate_type, candidate_abstract_numwords,
                     candidate_numattrs, mask_abstract, mask_query, mask_labels)
        loss = loss_fn(pred, label)
        loss.backward()

        #loss = loss_fn(pred, ner)
        optimizer.step()
        optimizer.zero_grad()
        # nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Clip gradients: gradients are modified in place
        train_loss += loss.item()

    train_loss = train_loss/len(train_part)

    model.eval()
    valid_loss = 0
    pred_set = []
    label_set = []
    for index, label, query, l_query, pos, candidate_abstract, l_abstract, \
        candidate_labels, l_labels, candidate_type, candidate_abstract_numwords, \
        candidate_numattrs in tqdm(valid_dataloader):
        model.zero_grad()
        query = nn.utils.rnn.pad_sequence(query, batch_first=True).type(torch.LongTensor).to(device)
        l_query = l_query.to(device)
        mask_query = get_mask(query, l_query, is_cuda=use_cuda).to(device).type(torch.float)

        candidate_abstract = nn.utils.rnn.pad_sequence(candidate_abstract, batch_first=True).type(torch.LongTensor).to(
            device)
        l_abstract = l_abstract.to(device)
        mask_abstract = get_mask(candidate_abstract, l_abstract, is_cuda=use_cuda).to(device).type(torch.float)

        candidate_labels = nn.utils.rnn.pad_sequence(candidate_labels, batch_first=True).type(torch.LongTensor).to(
            device)
        l_labels = l_labels.to(device)
        mask_labels = get_mask(candidate_labels, l_labels, is_cuda=use_cuda).to(device).type(torch.float)

        pos = pos.type(torch.LongTensor).to(device)

        candidate_type = candidate_type.to(device)
        label = label.type(torch.float).to(device).unsqueeze(1)
        candidate_numattrs = candidate_numattrs.to(device).type(torch.float)
        candidate_abstract_numwords = candidate_abstract_numwords.to(device).type(torch.float)
        # ner = ner.type(torch.float).cuda()
        # print(index)
        with torch.no_grad():
            pred = model(query, l_query, pos, candidate_abstract, l_abstract,
                     candidate_labels, l_labels, candidate_type, candidate_abstract_numwords,
                     candidate_numattrs, mask_abstract, mask_query, mask_labels)
        loss = loss_fn(pred, label)
        #print('loss',loss)
        pred_set.append(pred.cpu().numpy())
        label_set.append(label.cpu().numpy())
        valid_loss += loss.item()

    valid_loss = valid_loss / len(valid_part)
    pred_set = np.concatenate(pred_set, axis=0)
    label_set = np.concatenate(label_set, axis=0)
    INFO_THRE, thre_list = get_threshold(pred_set, label_set, num_feature=1)
    INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    logging.info(INFO + '\t' + INFO_THRE)
    print(INFO + '\t' + INFO_THRE)