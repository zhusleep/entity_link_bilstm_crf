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
from sklearn.externals import joblib

# file_namne = 'data/raw_data/train.json'
# train_part, valid_part = data_manager.read_entity_embedding(file_name=file_namne,train_num=10000000)
# seed_torch(2019)
#
# t = Tokenizer(max_feature=10000, segment=False, lowercase=True)
# corpus = [x[0] for x in train_part]+data_manager.read_eval()
# t.fit(corpus)

t = joblib.load('data/tokenizer.pkl')


train_part = data_manager.read_deep_cosine_test(filename='result/el_dev_result(1).csv')
seed_torch(2019)
print('train size %d', (len(train_part)))

# 准备embedding数据
embedding_file = 'embedding/miniembedding_baike_entity_vector.npy'
# embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'

if os.path.exists(embedding_file):
    embedding_matrix = np.load(embedding_file)
else:
    embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
    # embedding = '/home/zhu/Desktop/word_embedding/sgns.baidubaike.bigram-char'
    # embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
    embedding_matrix = load_glove(embedding, t.num_words + 100, t)
    np.save(embedding_file, embedding_matrix)

print('一共有%d 个字' % t.num_words)
data_all = np.array(train_part)
pred_vector = []
round = 0

train_dataset = Entity_Vector([x[0] for x in train_part], t, pos=[x[1] for x in train_part],
                              vector=[x[2] for x in train_part], label=[x[3] for x in train_part])

train_batch_size = 1024
valid_batch_size = 1024

model = EntityLink_entity_vector(vocab_size=embedding_matrix.shape[0], init_embedding=embedding_matrix,
                   encoder_size=128, dropout=0.2, num_outputs=len(data_manager.type_list))
for round in range(5):
    model.load_state_dict(torch.load('entity_embedding/gensim_vector_model_round %s'% (round)))
    use_cuda=True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)

    valid_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_link_entity_vector, shuffle=False, batch_size=valid_batch_size)

    #loss_fn = nn.CrossEntropyLoss()

    loss_fn = nn.MSELoss()
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
        with torch.no_grad():
            pred = model(X, mask_X, pos, vector, length)
        # print('loss',loss)

        pred_set.append(pred.cpu().numpy())
        # label_set.append(type.cpu().numpy())

    pred_set = np.concatenate(pred_set, axis=0)
    pred_vector.append(pred_set)
    # INFO_THRE, thre_list = get_threshold(pred_set, label_set, len(data_manager.type_list))
    # INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    # logging.info(INFO + '\t' + INFO_THRE)
pred_vector = np.concatenate(pred_vector, axis=1)
np.save('entity_embedding/gensim_vector_predict.npy', pred_vector)
    # 19 train loss　0.128424, val loss 0.153328, val_cos_loss 91155.319716
