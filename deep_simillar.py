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
from sklearn.model_selection import KFold, train_test_split


file_namne = 'data/raw_data/train.json'
data_all = data_manager.parse_v2(file_name=file_namne, valid_num=10000)
seed_torch(2019)

BERT_MODEL = 'bert-base-chinese'
CASED = False
t = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=True,
    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
#    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
)

kfold = KFold(n_splits=5, shuffle=False, random_state=2019)
pred_vector = []
round = 0
data_all = data_all[0:10000]
for train_index, test_index in kfold.split(np.zeros(len(data_all))):
    train_part = [data_all[i] for i in train_index]
    valid_part = [data_all[i] for i in test_index]

    train_dataset = entity_linking_v2(train_part, t, max_len=510)
    valid_dataset = entity_linking_v2(valid_part, t, max_len=510)

    train_batch_size = 128
    valid_batch_size = 128

    model = EntityLink_v2(vocab_size=len(data_manager.type_list), word_embed_size=768, encoder_size=128, dropout=0.2)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_linking_v2, shuffle=False, batch_size=train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_linking_v2, shuffle=False, batch_size=valid_batch_size)

    epoch = 20
    t_total = int(epoch*len(train_part)/train_batch_size)
    optimizer = BertAdam([
                    {'params': model.LSTM.parameters()},
                    {'params': model.hidden.parameters()},
                    {'params': model.classify.parameters()},
                    {'params': model.span_extractor.parameters()},
                    {'params': model.bert.parameters(), 'lr': 1e-3}
                ],  lr=1e-3, warmup=0.05, t_total=t_total)
    optimizer = torch.optim.Adam(model.parameters())

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
        for index, label, query, pos, candidate_abstract, candidate_type in tqdm(train_dataloader):
            query = nn.utils.rnn.pad_sequence(query, batch_first=True).type(torch.LongTensor).to(device)
            # l_query = l_query.to(device)
            # mask_query = get_mask(query, l_query, is_cuda=use_cuda).to(device).type(torch.float)

            candidate_abstract = nn.utils.rnn.pad_sequence(candidate_abstract, batch_first=True).type(torch.LongTensor).to(device)
            # l_abstract = l_abstract.to(device)
            # mask_abstract = get_mask(candidate_abstract, l_abstract, is_cuda=use_cuda).to(device).type(torch.float)

            pos = pos.type(torch.LongTensor).to(device)

            candidate_type = candidate_type.to(device)
            label = label.type(torch.float).to(device).unsqueeze(1)

            #ner = ner.type(torch.float).cuda()
            #print(index)
            pred = model(query, pos, candidate_abstract, candidate_type)
            loss = loss_fn(pred, label)
            loss.backward()

            #loss = loss_fn(pred, ner)
            optimizer.step()
            optimizer.zero_grad()
            # nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Clip gradients: gradients are modified in place
            train_loss += loss.item()*len(label)
            # break
        train_loss = train_loss/len(train_part)
        print(train_loss)
        model.eval()
        valid_loss = 0
        pred_set = []
        label_set = []
        for index, label, query, pos, candidate_abstract, candidate_type in tqdm(valid_dataloader):
            query = nn.utils.rnn.pad_sequence(query, batch_first=True).type(torch.LongTensor).to(device)
            # l_query = l_query.to(device)
            # mask_query = get_mask(query, l_query, is_cuda=use_cuda).to(device).type(torch.float)

            candidate_abstract = nn.utils.rnn.pad_sequence(candidate_abstract, batch_first=True).type(
                torch.LongTensor).to(device)
            # l_abstract = l_abstract.to(device)
            # mask_abstract = get_mask(candidate_abstract, l_abstract, is_cuda=use_cuda).to(device).type(torch.float)

            pos = pos.type(torch.LongTensor).to(device)

            candidate_type = candidate_type.to(device)
            label = label.type(torch.float).to(device).unsqueeze(1)

            with torch.no_grad():
                pred = model(query, pos, candidate_abstract, candidate_type)
            #print(pred.size(),type.size(),torch.max(type))

            loss = loss_fn(pred, label)
            #print('loss',loss)
            pred_set.append(pred.cpu().numpy())
            label_set.append(label.cpu().numpy())
            valid_loss += loss.item()*len(label)
        valid_loss = valid_loss / len(valid_part)
        pred_set = np.concatenate(pred_set, axis=0)
        label_set = np.concatenate(label_set, axis=0)
        print(np.mean(pred_set),np.mean(label_set))
        top_class = np.argmax(pred_set, axis=1)
        equals = top_class == label_set
        accuracy = np.mean(equals)
        print('acc', accuracy)
        print('train loss　%f, val loss %f'% (train_loss, valid_loss))