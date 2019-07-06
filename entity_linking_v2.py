#-*-coding:utf-8-*-
import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager,read_kb
from spo_dataset import entity_linking_v2, get_mask, collate_fn_linking_v2
from spo_model import SPOModel, EntityLink_v2
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

BERT_MODEL = 'bert-base-chinese'
CASED = False
t = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=True,
    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
#    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
)

train_dataset = entity_linking_v2(train_part, t, max_len=450)
valid_dataset = entity_linking_v2(valid_part, t, max_len=450)


train_batch_size = 16
valid_batch_size = 16

model = EntityLink_v2(encoder_size=128, dropout=0.2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_linking_v2, shuffle=True, batch_size=train_batch_size)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_linking_v2, shuffle=True, batch_size=valid_batch_size)

epoch = 20
t_total = int(epoch*len(train_part)/train_batch_size)
optimizer = BertAdam([
                {'params': model.LSTM.parameters()},
                {'params': model.hidden.parameters()},
                {'params': model.classify.parameters()},
                {'params': model.span_extractor.parameters()},
                {'params': model.bert.parameters(), 'lr': 2e-5}
            ],  lr=1e-3, warmup=0.05,t_total=t_total)
clip = 50

loss_fn = nn.BCELoss()
use_cuda=True
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
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
        # break
    train_loss = train_loss/len(train_part)

    model.eval()
    valid_loss = 0
    pred_set = []
    label_set = []
    for index, X, type, pos, length in tqdm(valid_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        length = length.cuda()
        mask_X = get_mask(X, length, is_cuda=True).cuda()
        pos = pos.type(torch.LongTensor).cuda()
        type = type.cuda()

        with torch.no_grad():
            pred = model(X, mask_X, pos, length)
        #print(pred.size(),type.size(),torch.max(type))

        loss = loss_fn(pred, type)
        #print('loss',loss)
        pred_set.append(pred.cpu().numpy())
        label_set.append(type.cpu().numpy())
        valid_loss += loss.item()
    valid_loss = valid_loss / len(valid_part)
    pred_set = np.concatenate(pred_set, axis=0)
    label_set = np.concatenate(label_set, axis=0)
    top_class = np.argmax(pred_set, axis=1)
    equals = top_class == label_set
    accuracy = np.mean(equals)
    print('acc', accuracy)
    print('train loss　%f, val loss %f'% (train_loss, valid_loss))