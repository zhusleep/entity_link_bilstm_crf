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
from utils import seed_torch, read_data, load_glove, calc_f1,get_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
import logging
import time
from torch.nn import functional as F
from sklearn.model_selection import KFold, train_test_split



file_namne = 'data/raw_data/train.json'
train_part, valid_part = data_manager.read_entity_embedding(file_name=file_namne,train_num=10000000)
seed_torch(2019)
print('train size %d, valid size %d', (len(train_part), len(valid_part)))

t = Tokenizer(max_feature=10000, segment=False, lowercase=True)
corpus = [x[0] for x in train_part]
t.fit(corpus)
# 准备embedding数据
embedding_file = 'embedding/miniembedding_baike_entity_vector.npy'
# embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'

if os.path.exists(embedding_file):
    embedding_matrix = np.load(embedding_file)
else:
    #embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
    embedding = '/home/zhu/Desktop/word_embedding/sgns.baidubaike.bigram-char'
    # embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
    embedding_matrix = load_glove(embedding, t.num_words + 100, t)
    np.save(embedding_file, embedding_matrix)

print('一共有%d 个字' % t.num_words)
data_all = np.array(train_part)
kfold = KFold(n_splits=2, shuffle=False, random_state=2019)
pred_vector = []
round = 0
for train_index, test_index in kfold.split(np.zeros(len(train_part))):
    train_part = data_all[train_index]
    valid_part = data_all[test_index]

    train_dataset = Entity_Vector([x[0] for x in train_part], t, pos=[x[1] for x in train_part],
                                  vector=[x[2] for x in train_part], label=[x[3] for x in train_part])
    valid_dataset = Entity_Vector([x[0] for x in valid_part], t, pos=[x[1] for x in valid_part],
                                  vector=[x[2] for x in valid_part], label=[x[3] for x in valid_part])

    train_batch_size = 1024
    valid_batch_size = 1024

    model = EntityLink_entity_vector(vocab_size=embedding_matrix.shape[0], init_embedding=embedding_matrix,
                       encoder_size=128, dropout=0.2, num_outputs=len(data_manager.type_list))
    use_cuda=True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)

    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_link_entity_vector, shuffle=False, batch_size=valid_batch_size)

    epoch = 10
    t_total = int(epoch*len(train_part)/train_batch_size)
    # optimizer = BertAdam([
    #                 {'params': model.LSTM.parameters()},
    #                 {'params': model.hidden.parameters()},
    #                 {'params': model.classify.parameters()},
    #                 {'params': model.span_extractor.parameters()},
    #                 {'params': model.bert.parameters(), 'lr': 2e-5}
    #             ],  lr=1e-3, warmup=0.05,t_total=t_total)
    optimizer = torch.optim.Adam(model.parameters())

    clip = 50

    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CosineEmbeddingLoss(margin=0)
    for epoch in range(epoch):
        print('round', round, 'epoch', epoch)
        model.train()
        train_loss = 0
        torch.cuda.manual_seed_all(epoch)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_link_entity_vector, shuffle=True, batch_size=train_batch_size)
        for index, X, label, pos, vector, length in tqdm(train_dataloader):
            #model.zero_grad()
            X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
            X = X.to(device)

            vector = vector.to(device).type(torch.float)
            length = length.to(device)
            #ner = ner.type(torch.float).cuda()
            mask_X = get_mask(X, length, is_cuda=use_cuda).to(device)
            pos = pos.type(torch.LongTensor).to(device)
            label = label.to(device).type(torch.float)

            pred = model(X, mask_X, pos, vector, length)
            loss = loss_fn(pred, vector, target=label)
            loss.backward()

            #loss = loss_fn(pred, ner)
            optimizer.step()
            optimizer.zero_grad()
            # Clip gradients: gradients are modified in place
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            train_loss += loss.item()
        train_loss = train_loss/len(train_part)*1e5

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
            loss = loss_fn(pred, vector, target=label)
            #cos_loss = torch.sum(F.cosine_similarity(pred, vector)).item()
            # print('loss',loss)

            pred_set.append(pred.cpu().numpy())
            # label_set.append(type.cpu().numpy())
            valid_loss += loss.item()

        valid_loss = valid_loss / len(valid_part)*1e5
        valid_cosloss = valid_cosloss/len(valid_part)*1e5
        pred_set = np.concatenate(pred_set, axis=0)
        # label_set = np.concatenate(label_set, axis=0)
        # top_class = np.argmax(pred_set, axis=1)
        # equals = top_class == label_set
        # accuracy = np.mean(equals)
        # print('acc', accuracy)
        print('train loss　%f, val loss %f ' % (train_loss, valid_loss))
    torch.save(model.state_dict(), 'entity_embedding/gensim_vector_model_round %s'% (round))
    pred_vector.append(pred_set)
    round += 1
    # INFO_THRE, thre_list = get_threshold(pred_set, label_set, len(data_manager.type_list))
    # INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    # logging.info(INFO + '\t' + INFO_THRE)
pred_vector = np.concatenate(pred_vector, axis=0)
np.save('entity_embedding/gensim_vector.npy', pred_vector)
    # 19 train loss　0.128424, val loss 0.153328, val_cos_loss 91155.319716
