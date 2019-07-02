import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager,read_kb
from spo_dataset import SPO_BERT, get_mask, collate_fn_withchar, collate_fn
from spo_model import SPOModel, SPO_Model_Bert
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1
from pytorch_pretrained_bert import BertTokenizer,BertAdam
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
## deal for bert
train_X = [['[CLS]']+list(temp)+['[SEP]'] for temp in train_X]
dev_X = [['[CLS]']+list(temp)+['[SEP]'] for temp in dev_X]
train_ner = [[0]+temp+[0]for temp in train_ner]
dev_ner = [[0]+temp+[0] for temp in dev_ner]


BERT_MODEL = 'bert-base-chinese'
CASED = False
t = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=True,
    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
#    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
)


train_dataset = SPO_BERT(train_X, t,  ner=train_ner)
valid_dataset = SPO_BERT(dev_X, t, ner=dev_ner)

batch_size = 2


# # 准备embedding数据
# #embedding_file = 'embedding/miniembedding_baike.npy'
# embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'
#
# if os.path.exists(embedding_file):
#     embedding_matrix = np.load(embedding_file)
# else:
#     #embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
#     #embedding = '/home/zhu/Desktop/word_embedding/sgns.baidubaike.bigram-char'
#     embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
#     embedding_matrix = load_glove(embedding, t.num_words+100, t)
#     np.save(embedding_file, embedding_matrix)

model = SPO_Model_Bert(encoder_size=128, dropout=0.5, num_tags=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, shuffle=False, batch_size=batch_size)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

epoch = 20
t_total = int(epoch*len(train_X)/batch_size)
optimizer = BertAdam([
                {'params': model.LSTM.parameters()},
                {'params': model.hidden.parameters()},
                {'params': model.NER.parameters()},
                {'params': model.crf_model.parameters()},
                {'params': model.bert.parameters(), 'lr': 1e-6}
            ],  lr=1e-3, warmup=0.05,t_total=t_total)
clip = 50


for epoch in range(epoch):
    model.train()
    train_loss = 0
    for index, X, ner, length in tqdm(train_dataloader):
        #model.zero_grad()
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        length = length.cuda()
        #ner = ner.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=True).cuda()
        ner = nn.utils.rnn.pad_sequence(ner, batch_first=True).type(torch.LongTensor)
        ner = ner.cuda()

        loss = model.cal_loss(X, mask_X, length, label=ner)
        loss.backward()

        #loss = loss_fn(pred, ner)
        optimizer.step()
        optimizer.zero_grad()
        # Clip gradients: gradients are modified in place
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss/len(train_X)

    model.eval()
    valid_loss = 0
    pred_set = []
    label_set = []
    for index, X, ner, length in tqdm(valid_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        length = length.cuda()
        mask_X = get_mask(X, length, is_cuda=True).cuda()
        label = ner
        ner = nn.utils.rnn.pad_sequence(ner, batch_first=True).type(torch.LongTensor)
        ner = ner.cuda()
        with torch.no_grad():
            pred = model(X, mask_X, length)
            loss = model.cal_loss(X, mask_X, length, label=ner)
        for i, item in enumerate(pred):
            pred_set.append(item[0:length.cpu().numpy()[i]])
        #pred_set.extend(pred)
        for item in label:
            label_set.append(item.numpy())
        valid_loss += loss.item()
    valid_loss = valid_loss/len(dev_X)

    acc,recall,f1,pred_result,label_result = calc_f1(pred_set, label_set, dev_X, data_manager.ner_list)
    INFO = 'epoch %d, train loss %f, valid loss %f, acc %f, recall %f, f1 %f '% (epoch, train_loss, valid_loss,acc,recall,f1)
    logging.info(INFO)
    print(INFO)
    #print(INFO+'\t'+INFO_THRE)

# 正负样本分析
dev = pd.DataFrame(dev_X,columns=['text'])
pred_mention = []
label_mention = []
for index,row in dev.iterrows():
    temp = []
    for item in pred_result[index]:
        temp.append(''.join(row['text'][item[0]:item[1]]))
    pred_mention.append(temp)
    temp = []
    for item in label_result[index]:
        temp.append(''.join(row['text'][item[0]:item[1]]))
    label_mention.append(temp)
dev['pred'] = pred_mention
dev['label'] = label_mention
dev.to_csv('result/analysis.csv')