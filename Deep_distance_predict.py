#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

import torch, os
from data_prepare import data_manager,read_kb
from spo_dataset import deep_distance_dataset, get_mask, collate_fn_linking_deep_distance
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
from sklearn.metrics import mean_squared_error as mse
from scipy.spatial.distance import cosine as cos

file_name = 'data/step1_test.pkl'
data_all = data_manager.deep_distance_test(file_name)
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
# data_all = data_all[0:10000]


valid_dataset = deep_distance_dataset(data_all, t, max_len=510)

valid_batch_size = 128
bert_model = 'bert-base-chinese'
from pytorch_pretrained_bert.modeling import BertModel

model = BertModel.from_pretrained(bert_model)
span_extractor = EndpointSpanExtractor(768)

valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_linking_deep_distance, shuffle=False, batch_size=valid_batch_size)

use_cuda=True
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
model.to(device)
span_extractor.to(device)
span_extractor.eval()
model.eval()
valid_loss = 0
point_distance = []
cos_distance = []
mse_distance = []
label_set = []
for index, label, query, pos, candidate_abstract, pos_answer in tqdm(valid_dataloader):
    query = nn.utils.rnn.pad_sequence(query, batch_first=True).type(torch.LongTensor).to(device)
    # l_query = l_query.to(device)
    # mask_query = get_mask(query, l_query, is_cuda=use_cuda).to(device).type(torch.float)

    candidate_abstract = nn.utils.rnn.pad_sequence(candidate_abstract, batch_first=True).type(
        torch.LongTensor).to(device)
    # l_abstract = l_abstract.to(device)
    # mask_abstract = get_mask(candidate_abstract, l_abstract, is_cuda=use_cuda).to(device).type(torch.float)

    pos = pos.type(torch.LongTensor).to(device)
    pos_answer = pos_answer.type(torch.LongTensor).to(device)

    label = label.type(torch.float).to(device).unsqueeze(1)

    with torch.no_grad():
        query_bert_outputs, _ = model(query, attention_mask=(query > 0).long(),
                                          token_type_ids=None,
                                          output_all_encoded_layers=True)
        query_bert_outputs = torch.cat(query_bert_outputs[-1:], dim=-1)
        pred = span_extractor(
            query_bert_outputs,
            pos
        ).squeeze(0)
        candidate_abstract_output, _ = model(candidate_abstract, attention_mask=(candidate_abstract > 0).long(),
                                      token_type_ids=None,
                                      output_all_encoded_layers=True)
        abstract_bert_outputs = torch.cat(candidate_abstract_output[-1:], dim=-1)
        label = span_extractor(
            abstract_bert_outputs,
            pos_answer
        ).squeeze(0)
    #print(pred.size(),type.size(),torch.max(type))
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    for i in range(query.size()[0]):
        mse_distance.append(mse(pred[i],label[i]))
        point_distance.append(sum(pred[i]*label[i]))
        cos_distance.append(cos(pred[i],label[i]))
    #print('loss',loss)

#pred_set = np.concatenate(pred_set, axis=0)
#label_set = np.concatenate(label_set, axis=0)
bert_dist = pd.DataFrame()
bert_dist['bert_cos_distance'] = cos_distance
bert_dist['bert_point_distance'] = point_distance
bert_dist['bert_mse_distance'] = mse_distance
bert_dist.to_pickle('data/bert_dis_test.pkl')