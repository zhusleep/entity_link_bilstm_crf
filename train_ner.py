#-*-coding:utf-8-*-


import torch
import pandas as pd
import numpy as np
from torchcrf import CRF

import torch, os
from spo_dataset import SPO, get_mask, collate_fn_ner
from spo_model import SPOModel, SPONerModel
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, get_threshold, sequence_cross_entropy_with_logits
import logging
import time
from torchcrf import CRF
from sklearn.metrics import classification_report

current_name = 'log/%s.txt' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.basicConfig(filename=current_name,
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


seed_torch(2019)
train_X, train_pos, train_label, train_ner, dev_X, dev_pos, dev_label, dev_ner = read_data()
assert len(train_X) == len(train_label)
# train_X = [''.join(word) for word in train_X]
# train_pos = [' '.join(word) for word in train_pos]

# dev_X = [' '.join(word) for word in dev_X]self.raw_X[index]
# dev_pos = [' '.join(word) for word in dev_pos]

t = Tokenizer(max_feature=500000, segment=False)
t.fit(list(train_X) + list(dev_X))

pos_t = Tokenizer(max_feature=500, segment=False)
pos_t.fit(list(train_pos)+list(dev_pos))
print('一共有%d 个词' % t.num_words)
print('一共有%d 个词性' % pos_t.num_words)

train_dataset = SPO(train_X, train_pos, train_label, t, pos_t, ner=train_ner)
valid_dataset = SPO(dev_X, dev_pos, dev_label, t, pos_t, ner=dev_ner)
batch_size = 512


# 准备embedding数据
embedding_file = 'embedding/miniembedding_engineer_baike_zi_new.npy'
#embedding_file = 'embedding/miniembedding_engineer_qq.npy'

if os.path.exists(embedding_file):
    embedding_matrix = np.load(embedding_file)
else:
    embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
    #embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
    embedding_matrix = load_glove(embedding, t.num_words+100, t)
    np.save(embedding_file, embedding_matrix)

model = SPONerModel(vocab_size=embedding_matrix.shape[0], embed_size=300,
                    encoder_size=128, dropout=0.5,
                 seq_dropout=0.0, init_embedding=embedding_matrix, dim_num_feat=1,
                 pos_embed_size=pos_t.num_words+100, pos_dim=10, ner_num=29)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_ner, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_ner, shuffle=False, batch_size=batch_size)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
crf = CRF(29, batch_first=True)
crf.to(device)


def ner_parser(pred, true):
    hit = sum((true==pred)&(true!=0))
    acc = hit/sum(pred!=0)
    recall = hit/sum(true!=0)
    print(classification_report(true, pred))
    INFO = 'acc %f, recall%f, f1% f' % (acc, recall, 2*acc*recall/(acc+recall))
    return INFO


# 30万词由词性平均向量替换未知向量
def cal_entity(seq):
    e = []
    for i in range(len(seq)):
        if seq[i] != 0 and i==0:
            e.append(seq[i])
        elif seq[i] != 0 and seq[i-1]!=seq[i]:
            e.append(seq[i])
    return e


for epoch in range(25):
    model.train()
    train_loss = 0
    for index, X, pos_tags, length, numerical_features, ner, intent_label in tqdm(train_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        pos_tags = nn.utils.rnn.pad_sequence(pos_tags, batch_first=True).type(torch.LongTensor)
        pos_tags = pos_tags.cuda()
        length = length.cuda()
        n_feats = numerical_features.type(torch.float).cuda()
        ner = nn.utils.rnn.pad_sequence(ner, batch_first=True).type(torch.LongTensor)
        ner = ner.cuda()
        intent_label = intent_label.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=True)
        ###################
        ner_logits, ner_pred, intent_pred = model(X, pos_tags, mask_X, length, n_feats)

        #loss_slot = -crf(ner_logits, ner, mask=mask_X)

        loss1 = sequence_cross_entropy_with_logits(ner_logits, ner, mask_X,
                                                   label_smoothing=False)
        loss2 = loss_fn(intent_pred, intent_label)
        loss = loss1+loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss1.item()*X.size()[0]

        #break
    train_loss = train_loss/len(train_X)

    model.eval()
    valid_loss = 0
    ner_pred_set = []

    ner_label_set = []
    total_length = 0
    pred_n = 0
    recal_n = 0
    hit_n = 0
    for index, X, pos_tags, length, numerical_features, ner, intent_label in tqdm(valid_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        pos_tags = nn.utils.rnn.pad_sequence(pos_tags, batch_first=True).type(torch.LongTensor)
        pos_tags = pos_tags.cuda()
        length = length.cuda()
        n_feats = numerical_features.type(torch.float).cuda()
        ner = nn.utils.rnn.pad_sequence(ner, batch_first=True).type(torch.LongTensor)
        ner = ner.cuda()
        intent_label = intent_label.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=True)
        with torch.no_grad():
            ner_logits, ner_pred, intent_pred = model(X, pos_tags, mask_X, length, n_feats)
        loss = sequence_cross_entropy_with_logits(ner_logits, ner, mask_X,
                                                  label_smoothing=False)
        ner_pred = ner_pred.argmax(dim=-1)
        #ner_pred = crf.decode(ner_logits, mask=mask_X)
        for i, item in enumerate(ner_pred):
            x = ner_pred[i][0:length[i]].cpu().numpy()
            y = ner[i][0:length[i]].cpu().numpy()
            ner_pred_set.append(x)
            ner_label_set.append(y)
        intent_pred_set.append(intent_pred.cpu().numpy())
        intent_label_set.append(intent_label.cpu().numpy())
        # ner_pred_set.append(ner_pred.view(-1, ner_pred.size()[-1]).argmax(dim=-1).cpu().numpy())
        # ner_label_set.append(ner.view(-1).cpu().numpy())
        valid_loss += loss*X.size()[0]
        total_length += length.sum()
    valid_loss = valid_loss/len(dev_X)
    intent_pred_set = np.concatenate(intent_pred_set, axis=0)
    intent_label_set = np.concatenate(intent_label_set, axis=0)
    ner_pred_set = np.concatenate(ner_pred_set, axis=0)
    ner_label_set = np.concatenate(ner_label_set, axis=0)
    INFO_THRE = ner_parser(ner_pred_set, ner_label_set)
    #INFO_THRE = get_threshold(pred_set, label_set)
    INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    logging.info(INFO+'\t'+INFO_THRE)
    print(INFO+'\t'+INFO_THRE)
    INFO_THRE = get_threshold(intent_pred_set, intent_label_set)
    #INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    logging.info(INFO_THRE)
    print(INFO_THRE)





