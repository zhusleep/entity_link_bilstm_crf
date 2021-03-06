#-*-coding:utf-8-*-
import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager,read_kb
from spo_dataset import SPO_BERT_LINK, get_mask, collate_fn_link, get_mask_bertpiece
from spo_model import SPOModel, EntityLink_bert
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1,get_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
from sklearn.model_selection import KFold

import logging
import time
from sklearn.externals import joblib


file_namne = 'data/raw_data/train.json'
data_all = data_manager.parse_mention(file_name=file_namne, valid_num=10000)
print('一共有%d 个字' % len(data_all))
#data_all = np.array(data_all)
kfold = KFold(n_splits=5, shuffle=False, random_state=2019)
pred_vector = []
round = 0
train_batch_size = 2048
for train_index, test_index in kfold.split(np.zeros(len(data_all))):
    train_part = [data_all[i] for i in train_index]

    valid_part = []
    for i in test_index:
        if data_all[i][-1]!= 0:
            valid_part.append(data_all[i])
    #valid_part = [data_all[i] for i in test_index]


    BERT_MODEL = 'bert-base-chinese'
    CASED = False
    t = BertTokenizer.from_pretrained(
        BERT_MODEL,
        do_lower_case=True,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        #    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
        )
    use_cuda = True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = EntityLink_bert(encoder_size=128, dropout=0.2, num_outputs=len(data_manager.type_list))
    model.to(device)

    # [s['text_id'], sentence, pos, m_type]
    seed_torch(2019)
    train_X = [['[CLS]']+list(temp[1])+['[SEP]'] for temp in train_part]
    dev_X = [['[CLS]']+list(temp[1])+['[SEP]'] for temp in valid_part]
    train_pos = [[x[2][0]+1, x[2][1]+1]for x in train_part]
    dev_pos = [[x[2][0]+1, x[2][1]+1]for x in valid_part]
    train_type = [data_manager.type_list.index(x[3]) for x in train_part]
    dev_type = [data_manager.type_list.index(x[3]) for x in valid_part]

    print(max(train_type), min(train_type))
    # add kb
    # train_X = train_X+train_X_kb
    # train_pos += train_pos_kb
    # train_type += train_type_kb

    train_dataset = SPO_BERT_LINK(train_X, t, pos=train_pos, type=train_type)
    valid_dataset = SPO_BERT_LINK(dev_X, t, pos=dev_pos, type=dev_type)

    train_batch_size = 64
    valid_batch_size = 128

    #model.load_state_dict(torch.load('model_type/deep_type.pth'))

    use_cuda=True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_link, shuffle=False, batch_size=train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_link, shuffle=False, batch_size=valid_batch_size)

    epoch = 5
    t_total = int(epoch*len(train_X)/train_batch_size)
    optimizer = BertAdam([
                    {'params': model.LSTM.parameters()},
                    {'params': model.hidden.parameters()},
                    {'params': model.classify.parameters()},
                    {'params': model.span_extractor.parameters()},
                    {'params': model.bert.parameters(), 'lr': 2e-5}
                ],  lr=1e-3, warmup=0.05, t_total=t_total)
    clip = 50
    model.load_state_dict(torch.load('model_type/deep_type_t_%d.pth' % round))

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epoch):

        train_loss = 0
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
            mask_for_pool = get_mask_bertpiece(X, length, pos, is_cuda=True).type(torch.float)

            with torch.no_grad():
                pred = model(X, mask_X, pos, length, mask_for_pool).to(device)
            #print(pred.size(),type.size(),torch.max(type))

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
        print('train loss　%f, val loss %f' % (train_loss, valid_loss))
        break

    #torch.save(model.state_dict(), 'model_type/deep_type_t_%d.pth'%round)
    pred_vector.append(pred_set)
    round += 1
    # INFO_THRE, thre_list = get_threshold(pred_set, label_set, len(data_manager.type_list))
    # INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    # logging.info(INFO + '\t' + INFO_THRE)
pred_vector = np.concatenate(pred_vector, axis=0)
np.save('model_type/type_vector.npy', pred_vector)
# acc 0.7933060109289618
# train loss　0.015648, val loss 0.005099
# acc 0.7930327868852459


    # INFO_THRE, thre_list = get_threshold(pred_set, label_set, len(data_manager.type_list))
    # INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    # logging.info(INFO + '\t' + INFO_THRE)
    # print(INFO + '\t' + INFO_THRE)
# acc 0.7933704754896808
# train loss　0.008100, val loss 0.005171
# acc 0.8103998037772873
# train loss　0.004624, val loss 0.004680
# acc 0.8161638459651704
# train loss　0.003730, val loss 0.004623
#   0%|          | 0/1563 [00:00<?, ?it/s]acc 0.8191597463120642
# train loss　0.003022, val loss 0.004722
# acc 0.8223133256245839
# train loss　0.002412, val loss 0.004954
# acc 0.8238375556256351
# train loss　0.001904, val loss 0.005256
# acc 0.8233119590735485
# train loss　0.001499, val loss 0.005590
#   0%|          | 0/1563 [00:00<?, ?it/s]acc 0.8213146921756194
# train loss　0.001206, val loss 0.005952
# acc 0.8224359648200708
# train loss　0.000966, val loss 0.006328
#   0%|          | 0/1563 [00:00<?, ?it/s]acc 0.8232769193034094
# train loss　0.000773, val loss 0.006691

#
# 82187it [00:01, 46426.07it/s]0
# 90000it [00:01, 57428.81it/s]
# 100%|██████████| 3125/3125 [33:29<00:00,  1.52it/s]
# 100%|██████████| 892/892 [02:38<00:00,  6.01it/s]
# acc 0.8120291530887558
# train loss　0.014729, val loss 0.009327
# 100%|██████████| 3125/3125 [33:45<00:00,  1.56it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.86it/s]
# acc 0.8346298048284804
# train loss　0.008138, val loss 0.008352
# 100%|██████████| 3125/3125 [33:48<00:00,  1.59it/s]
# 100%|██████████| 892/892 [02:38<00:00,  6.18it/s]
# acc 0.8479974771365499
# train loss　0.006041, val loss 0.008101
# 100%|██████████| 3125/3125 [33:30<00:00,  1.54it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.53it/s]
#   0%|          | 0/3125 [00:00<?, ?it/s]acc 0.848347874837941
# train loss　0.004538, val loss 0.008875
# 100%|██████████| 3125/3125 [33:36<00:00,  1.53it/s]
# 100%|██████████| 892/892 [02:37<00:00,  6.11it/s]
# acc 0.8509933774834437
# train loss　0.003402, val loss 0.009551
# 100%|██████████| 3125/3125 [33:43<00:00,  1.56it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.73it/s]
# acc 0.849451627597323
# train loss　0.002501, val loss 0.010483
# 100%|██████████| 3125/3125 [33:46<00:00,  1.58it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.99it/s]
# acc 0.8536038403588072
# train loss　0.001909, val loss 0.011403
# 100%|██████████| 3125/3125 [33:39<00:00,  1.50it/s]
# 100%|██████████| 892/892 [02:37<00:00,  5.89it/s]
# acc 0.8509933774834437
# train loss　0.001453, val loss 0.012390
# 100%|██████████| 3125/3125 [33:32<00:00,  1.47it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.98it/s]
# acc 0.8536038403588072
# train loss　0.001165, val loss 0.012897
# 100%|██████████| 3125/3125 [33:32<00:00,  1.57it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.89it/s]
#   0%|          | 0/3125 [00:00<?, ?it/s]acc 0.8549703913942325
# train loss　0.000933, val loss 0.013812
# 100%|██████████| 3125/3125 [33:28<00:00,  1.58it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.86it/s]
# acc 0.8563369424296576
# train loss　0.000735, val loss 0.014768
# 100%|██████████| 3125/3125 [33:34<00:00,  1.54it/s]
# 100%|██████████| 892/892 [02:38<00:00,  6.00it/s]
# acc 0.8566523003609097
# train loss　0.000628, val loss 0.015229
# 100%|██████████| 3125/3125 [33:32<00:00,  1.49it/s]
# 100%|██████████| 892/892 [02:37<00:00,  5.85it/s]
# acc 0.8561617435789621
# train loss　0.000514, val loss 0.015833
# 100%|██████████| 3125/3125 [33:31<00:00,  1.53it/s]
# 100%|██████████| 892/892 [02:38<00:00,  6.04it/s]
# acc 0.8551981499001367
# train loss　0.000422, val loss 0.016027
# 100%|██████████| 3125/3125 [33:30<00:00,  1.53it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.69it/s]
#   0%|          | 0/3125 [00:00<?, ?it/s]acc 0.856950138407092
# train loss　0.000353, val loss 0.016648
# 100%|██████████| 3125/3125 [33:33<00:00,  1.61it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.95it/s]
# acc 0.8579137320859175
# train loss　0.000275, val loss 0.017072
# 100%|██████████| 3125/3125 [33:28<00:00,  1.63it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.95it/s]
# acc 0.8582816496723782
# train loss　0.000233, val loss 0.017453
# 100%|██████████| 3125/3125 [33:25<00:00,  1.56it/s]
# 100%|██████████| 892/892 [02:38<00:00,  5.74it/s]
#   0%|          | 0/3125 [00:00<?, ?it/s]acc 0.8591751638109254
# train loss　0.000193, val loss 0.017747
# 100%|██████████| 3125/3125 [33:25<00:00,  1.49it/s]
# 100%|██████████| 892/892 [02:37<00:00,  6.11it/s]
#   0%|          | 0/3125 [00:00<?, ?it/s]acc 0.8584743684081433
# train loss　0.000154, val loss 0.018126
# 100%|██████████| 3125/3125 [33:23<00:00,  1.55it/s]
# 100%|██████████| 892/892 [02:38<00:00,  6.15it/s]
# acc 0.859192683695995
# bert new model ###############################################
# 100%|██████████| 3203/3203 [34:15<00:00,  1.84it/s]
# 100%|██████████| 401/401 [02:23<00:00,  3.65it/s]
# acc 0.8252537080405933
# train loss　0.012043, val loss 0.004385
# 100%|██████████| 3203/3203 [34:53<00:00,  5.06s/it]
# 100%|██████████| 401/401 [02:27<00:00,  3.28it/s]
# acc 0.846272443403591
# train loss　0.007218, val loss 0.003969
# 100%|██████████| 3203/3203 [34:29<00:00,  1.76it/s]
# 100%|██████████| 401/401 [02:22<00:00,  3.61it/s]
# acc 0.8511124121779859
# train loss　0.005298, val loss 0.004034
# 100%|██████████| 3203/3203 [36:23<00:00,  1.61it/s]
# 100%|██████████| 401/401 [02:33<00:00,  3.18it/s]
# acc 0.8538056206088993
# train loss　0.003952, val loss 0.004224
# 100%|██████████| 3203/3203 [36:34<00:00,  1.69it/s]
# 100%|██████████| 401/401 [02:32<00:00,  3.32it/s]
# acc 0.8551522248243559
# train loss　0.003032, val loss 0.004376

