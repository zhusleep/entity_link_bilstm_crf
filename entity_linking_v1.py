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
train_X, train_pos, train_type, dev_X, dev_pos, dev_type = data_manager.parse_mention(file_name=file_namne,valid_num=10000)
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


train_batch_size = 1024
valid_batch_size = 1024

model = EntityLink(vocab_size=embedding_matrix.shape[0], init_embedding=embedding_matrix,
                   encoder_size=128, dropout=0.2, num_outputs=len(data_manager.type_list))
use_cuda=True
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
model.to(device)

valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_link, shuffle=False, batch_size=valid_batch_size)

epoch = 20
t_total = int(epoch*len(train_X)/train_batch_size)
# optimizer = BertAdam([
#                 {'params': model.LSTM.parameters()},
#                 {'params': model.hidden.parameters()},
#                 {'params': model.classify.parameters()},
#                 {'params': model.span_extractor.parameters()},
#                 {'params': model.bert.parameters(), 'lr': 2e-5}
#             ],  lr=1e-3, warmup=0.05,t_total=t_total)
optimizer = torch.optim.Adam({'params':model.word_embedding.parameters(),'lr':1e-4},
                             model.parameters())

clip = 50

loss_fn = nn.CrossEntropyLoss()
for epoch in range(epoch):
    model.train()
    train_loss = 0
    torch.cuda.manual_seed_all(epoch)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_link, shuffle=True, batch_size=train_batch_size)
    for index, X, type, pos, length in tqdm(train_dataloader):
        #model.zero_grad()
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.to(device)
        length = length.to(device)
        #ner = ner.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=use_cuda).to(device)
        pos = pos.type(torch.LongTensor).to(device)
        type = type.to(device)

        pred = model(X, mask_X, pos, length)
        loss = loss_fn(pred, type)
        loss.backward()

        #loss = loss_fn(pred, ner)
        optimizer.step()
        optimizer.zero_grad()
        # Clip gradients: gradients are modified in place
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        train_loss += loss.item()
    train_loss = train_loss/len(train_X)

    model.eval()
    valid_loss = 0
    pred_set = []
    label_set = []
    for index, X, type, pos, length in tqdm(valid_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.to(device)
        length = length.to(device)

        # ner = ner.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=use_cuda).to(device)
        pos = pos.type(torch.LongTensor).to(device)
        type = type.to(device)

        with torch.no_grad():
            pred = model(X, mask_X, pos, length)
        loss = loss_fn(pred, type)
        #print('loss',loss)

        pred_set.append(pred.cpu().numpy())
        label_set.append(type.cpu().numpy())
        valid_loss += loss.item()
    valid_loss = valid_loss / len(dev_X)
    pred_set = np.concatenate(pred_set, axis=0)
    label_set = np.concatenate(label_set, axis=0)
    top_class = np.argmax(pred_set, axis=1)
    equals = top_class == label_set
    accuracy = np.mean(equals)
    print('acc', accuracy)
    print('train loss　%f, val loss %f'% (train_loss, valid_loss))
    # INFO_THRE, thre_list = get_threshold(pred_set, label_set, len(data_manager.type_list))
    # INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    # logging.info(INFO + '\t' + INFO_THRE)
    # print(INFO + '\t' + INFO_THRE)