import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager
from spo_dataset import SPO, get_mask, collate_fn_withchar, collate_fn
from spo_model import SPOModel, SPO_Model_Simple
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1
import logging
import time

current_name = 'log/%s.txt' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.basicConfig(filename=current_name,
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


seed_torch(2019)
file_namne = 'data/raw_data/train.json'
train_X, train_ner, dev_X, dev_ner = data_manager.parseData(file_name=file_namne,valid_num=10000)
assert len(train_X) == len(train_ner)

t = Tokenizer(max_feature=10000, segment=False, lowercase=True)
t.fit(train_X+dev_X)

print('一共有%d 个字' % t.num_words)


train_dataset = SPO(train_X, t, max_len=50, ner=train_ner)
valid_dataset = SPO(dev_X, t, max_len=50, ner=dev_ner)

batch_size = 1024


# 准备embedding数据
embedding_file = 'embedding/miniembedding_baike.npy'
#embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'

if os.path.exists(embedding_file):
    embedding_matrix = np.load(embedding_file)
else:
    embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
    #embedding = '/home/zhu/Desktop/word_embedding/sgns.baidubaike.bigram-char'
    #embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
    embedding_matrix = load_glove(embedding, t.num_words+100, t)
    np.save(embedding_file, embedding_matrix)

model = SPO_Model_Simple(vocab_size=embedding_matrix.shape[0],
                 word_embed_size=embedding_matrix.shape[1], encoder_size=128, dropout=0.5,
                 seq_dropout=0.0, init_embedding=embedding_matrix)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, shuffle=False, batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters())
clip = 50

for epoch in range(25):
    model.train()
    train_loss = 0
    for index, X, ner, length in tqdm(train_dataloader):
        #model.zero_grad()
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        length = length.cuda()
        #ner = ner.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=True)
        ner = nn.utils.rnn.pad_sequence(ner, batch_first=True).type(torch.LongTensor)
        ner = ner.cuda()

        loss = model.cal_loss(X, mask_X, length, label=ner)
        loss.backward()

        #loss = loss_fn(pred, ner)
        optimizer.step()
        optimizer.zero_grad()
        # Clip gradients: gradients are modified in place
        #_ = nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        #break
    train_loss = train_loss/len(train_X)

    model.eval()
    valid_loss = 0
    pred_set = []
    label_set = []
    for index, X, ner, length in tqdm(valid_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        length = length.cuda()
        mask_X = get_mask(X, length, is_cuda=True)
        label = ner
        ner = nn.utils.rnn.pad_sequence(ner, batch_first=True).type(torch.LongTensor)
        ner.cuda()
        with torch.no_grad():
            pred = model(X, mask_X, length)
            loss = model.cal_loss(X, mask_X, length,label=ner)
        pred_set.extend(pred)
        for item in label:
            label_set.append(item.numpy())
        valid_loss += loss.item()
    valid_loss = valid_loss/len(dev_X)

    acc,recall,f1 = calc_f1(pred_set, label_set, data_manager.ner_list)
    INFO = 'epoch %d, train loss %f, valid loss %f, acc %f, recall %f, f1 %f '% (epoch, train_loss, valid_loss,acc,recall,f1)
    logging.info(INFO)
    print(INFO)
    #print(INFO+'\t'+INFO_THRE)


