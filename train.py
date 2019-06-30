import pandas as pd
import numpy as np

import torch, os
from spo_dataset import SPO, get_mask, collate_fn_withchar, collate_fn
from spo_model import SPOModel
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, get_threshold
import logging
import time

current_name = 'log/%s.txt' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.basicConfig(filename=current_name,
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


seed_torch(2019)
train_X, train_pos, train_label, train_ner, dev_X, dev_pos, dev_label, dev_ner = read_data(char=False)
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

combined = True
if combined:
    combined_char_t = Tokenizer(max_feature=10000, segment=False)
    vocab = [''.join(x) for x in list(train_X)+list(dev_X)]
    combined_char_t.fit(vocab)
    # 准备embedding数据
    embedding_file = 'embedding/miniembedding_engineer_baike_char.npy'
    # embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'

    if os.path.exists(embedding_file):
        char_embedding_matrix = np.load(embedding_file)
    else:
        embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
        # embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
        char_embedding_matrix = load_glove(embedding, combined_char_t.num_words + 100, t)
        np.save(embedding_file, char_embedding_matrix)
else:
    combined_char_t = None


train_dataset = SPO(train_X, train_pos, t, pos_t, label=train_label, combined_char_t=combined_char_t)
valid_dataset = SPO(dev_X, dev_pos, t, pos_t, label=dev_label, combined_char_t=combined_char_t)

batch_size = 40


# 准备embedding数据
embedding_file = 'embedding/miniembedding_engineer_baike_word.npy'
#embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'

if os.path.exists(embedding_file):
    embedding_matrix = np.load(embedding_file)
else:
    embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
    #embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
    embedding_matrix = load_glove(embedding, t.num_words+100, t)
    np.save(embedding_file, embedding_matrix)

model = SPOModel(vocab_size=embedding_matrix.shape[0], char_vocab_size=char_embedding_matrix.shape[0],
                 word_embed_size=embedding_matrix.shape[1], encoder_size=128, dropout=0.5,
                 seq_dropout=0.0, init_embedding=embedding_matrix, char_init_embedding=char_embedding_matrix,
                 dim_num_feat=1, pos_embed_size=pos_t.num_words+100, pos_dim=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_withchar, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_withchar, shuffle=False, batch_size=batch_size)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
clip = 50

for epoch in range(25):
    model.train()
    train_loss = 0
    for index, X, pos_tags, length, numerical_features, char_vocab, label in tqdm(train_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        pos_tags = nn.utils.rnn.pad_sequence(pos_tags, batch_first=True).type(torch.LongTensor)
        pos_tags = pos_tags.cuda()
        length = length.cuda()
        n_feats = numerical_features.type(torch.float).cuda()
        label = label.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=True).type(torch.float)

        pred = model(X, pos_tags, mask_X, length, n_feats, char_vocab)

        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
        #_ = nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        #break
    train_loss = train_loss/len(train_X)

    model.eval()
    valid_loss = 0
    pred_set = []
    label_set = []
    for index, X, pos_tags, length, numerical_features, char_vocab, label in tqdm(valid_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        pos_tags = nn.utils.rnn.pad_sequence(pos_tags, batch_first=True).type(torch.LongTensor)
        pos_tags = pos_tags.cuda()
        length = length.cuda()
        n_feats = numerical_features.type(torch.float).cuda()
        label = label.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=True).type(torch.float)
        with torch.no_grad():
            pred = model(X, pos_tags, mask_X, length, n_feats, char_vocab)
        loss = loss_fn(pred, label)
        pred_set.append(pred.cpu().numpy())
        label_set.append(label.cpu().numpy())
        valid_loss += loss
    valid_loss = valid_loss/len(dev_X)
    pred_set = np.concatenate(pred_set, axis=0)
    label_set = np.concatenate(label_set, axis=0)
    INFO_THRE = get_threshold(pred_set, label_set)
    INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    logging.info(INFO+'\t'+INFO_THRE)
    print(INFO+'\t'+INFO_THRE)


