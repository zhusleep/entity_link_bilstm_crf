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
train_X, train_pos, train_type, dev_X, dev_pos, dev_type = data_manager.parse_v2(file_name=file_namne, valid_num=10000)
seed_torch(2019)