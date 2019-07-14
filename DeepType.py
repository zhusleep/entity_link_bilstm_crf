#-*-coding:utf-8-*-
import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager,read_kb
from spo_dataset import SPO_BERT_LINK, get_mask, collate_fn_link
from spo_model import SPOModel, EntityLink_bert
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1,get_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
from sklearn.model_selection import KFold, train_test_split
import pickle
from sklearn.externals import joblib

if not os.path.exists('data/data_enhance_backup.pkl'):
    data_corpus = data_manager.data_enhance(max_len=50)
    joblib.dump(data_manager, 'data/data_enhance_backup.pkl')

else:
    data_manager = joblib.load('data/data_enhance_backup.pkl')
    data_corpus = data_manager.enhanced_data
# data_corpus = pickle.load(open('data/data_enhance.pkl', 'rb'))

kfold = KFold(n_splits=1000, shuffle=False, random_state=2019)
pred_vector = []
round = 0
for train_index, valid_index in kfold.split(np.zeros(len(data_corpus))):
    train_data = [data_corpus[i] for i in train_index]
    valid_data = [data_corpus[i] for i in valid_index]

    #file_namne = 'data/raw_data/train.json'
    #train_X, train_pos, train_type, dev_X, dev_pos, dev_type =
    seed_torch(2019)
    train_X = [['[CLS]']+list(temp[0])+['[SEP]'] for temp in train_data]
    dev_X = [['[CLS]']+list(temp[0])+['[SEP]'] for temp in valid_data]
    train_pos = [[x[1]+1, x[2]+1] for x in train_data]
    dev_pos = [[x[1]+1, x[2]+1]for x in valid_data]
    train_type = [x[3] for x in train_data]
    valid_type = [x[3] for x in valid_data]


    BERT_MODEL = 'bert-base-chinese'
    CASED = False
    t = BertTokenizer.from_pretrained(
        BERT_MODEL,
        do_lower_case=True,
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
    #    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
    )

    train_dataset = SPO_BERT_LINK(train_X, t, pos=train_pos, type=train_type, max_len=510)
    valid_dataset = SPO_BERT_LINK(dev_X, t, pos=dev_pos, type=valid_type, max_len=510)

    train_batch_size = 128
    valid_batch_size = 128

    model = EntityLink_bert(encoder_size=128, dropout=0.2, num_outputs=len(data_manager.type_list))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_link, shuffle=True, batch_size=train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_link, shuffle=True, batch_size=valid_batch_size)

    epoch = 20
    t_total = int(epoch*len(train_X)/train_batch_size)
    optimizer = BertAdam([
                    {'params': model.LSTM.parameters()},
                    {'params': model.hidden.parameters()},
                    {'params': model.classify.parameters()},
                    {'params': model.span_extractor.parameters()},
                    {'params': model.bert.parameters(), 'lr': 2e-5}
                ],  lr=1e-3, warmup=0.05, t_total=t_total)
    clip = 50

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epoch):
        model.train()
        train_loss = 0
        for index, X, type, pos, length in tqdm(train_dataloader):
            #model.zero_grad()
            X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
            X = X.cuda()
            length = length.cuda()
            #ner = ner.type(torch.float).cuda()
            mask_X = get_mask(X, length, is_cuda=True).cuda()
            pos = pos.type(torch.LongTensor).cuda()
            type = type.cuda()
            #print(index)
            pred = model(X, mask_X, pos, length)
            loss = loss_fn(pred, type)
            loss.backward()

            #loss = loss_fn(pred, ner)
            optimizer.step()
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Clip gradients: gradients are modified in place
            train_loss += loss.item()
            #break
        train_loss = train_loss/len(train_X)

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
            #break
        valid_loss = valid_loss / len(dev_X)
        pred_set = np.concatenate(pred_set, axis=0)
        label_set = np.concatenate(label_set, axis=0)
        top_class = np.argmax(pred_set, axis=1)
        equals = top_class == label_set
        accuracy = np.mean(equals)
        print('acc', accuracy)
        print('train loss　%f, val loss %f'% (train_loss, valid_loss))
        if epoch==2:
            break
    torch.save(model.state_dict(), 'model_type/deep_type.pth')
    break

        # INFO_THRE, thre_list = get_threshold(pred_set, label_set, len(data_manager.type_list))
        # INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
        # logging.info(INFO + '\t' + INFO_THRE)
    # print(INFO + '\t' + INFO_THRE)