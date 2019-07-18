#-*-coding:utf-8-*-

from qapair_model import QAModel
from torch.autograd import Variable
import numpy as np
import torch, os
from data_prepare import data_manager,read_kb
from spo_dataset import QAPair, get_mask, qapair_collate_fn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1,get_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
from sklearn.model_selection import KFold


file_namne = 'data/raw_data/train.json'
data_all = data_manager.read_deep_match()
seed_torch(2019)

t = Tokenizer(max_feature=10000, segment=False, lowercase=True)
corpus = list(data_all['question'])+list(data_all['answer'])
t.fit(corpus)
# 准备embedding数据
embedding_file = 'embedding/miniembedding_baike_deepmatch.npy'
# embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'

if os.path.exists(embedding_file):
    embedding_matrix = np.load(embedding_file)
else:
    embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
    #embedding = '/home/zhu/Desktop/word_embedding/sgns.baidubaike.bigram-char'
    # embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
    embedding_matrix = load_glove(embedding, t.num_words + 100, t)
    np.save(embedding_file, embedding_matrix)


print('一共有%d 个字' % t.num_words)
#data_all = np.array(data_all)
kfold = KFold(n_splits=5, shuffle=False, random_state=2019)
pred_vector = []
round = 0
train_batch_size = 2048
for train_index, test_index in kfold.split(np.zeros(len(data_all))):
    model = QAModel(vocab_size=embedding_matrix.shape[0], embed_size=300, encoder_size=64, dropout=0.5,
                    seq_dropout=0.2, init_embedding=embedding_matrix, dim_num_feat=346)

    use_cuda = True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)

    train_part = data_all.loc[train_index,:]
    valid_part = data_all.loc[test_index,:]

    train_dataset = QAPair(train_part, t, data_manager.predictor)
    valid_dataset = QAPair(valid_part, t, data_manager.predictor)

    train_dataloader = DataLoader(train_dataset, collate_fn=qapair_collate_fn, shuffle=True, batch_size=train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=qapair_collate_fn, shuffle=False, batch_size=train_batch_size)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(3):
        model.train()
        model.to(device)
        train_loss = 0
        for sa, sb, la, lb, numerical_features, label in tqdm(train_dataloader):
            sa = nn.utils.rnn.pad_sequence(sa, batch_first=True).type(torch.LongTensor)
            sb = nn.utils.rnn.pad_sequence(sb, batch_first=True).type(torch.LongTensor)
            n_feats = numerical_features.type(torch.FloatTensor)
            sa = Variable(sa.to(device))
            sb = Variable(sb.to(device))
            la = la.to(device)
            lb = lb.to(device)
            n_feats = Variable(n_feats.to(device))
            label = label.type(torch.FloatTensor).view(-1, 1).to(device)
            mask_a = get_mask(sa, la, is_cuda=use_cuda)
            mask_b = get_mask(sb, lb, is_cuda=use_cuda)

            pred = model(sa, sb, mask_a, mask_b, la, lb, n_feats)
            loss = loss_fn(pred, label)
            train_loss += loss.item()*len(sa)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #break
        model.eval()
        valid_loss = 0
        trn_labels, trn_preds = [], []
        for sa, sb, la, lb, numerical_features, label in valid_dataloader:
            sa = nn.utils.rnn.pad_sequence(sa, batch_first=True).type(torch.LongTensor)
            sb = nn.utils.rnn.pad_sequence(sb, batch_first=True).type(torch.LongTensor)
            n_feats = numerical_features.type(torch.FloatTensor)
            sa = Variable(sa.cuda()).squeeze(-1)
            sb = Variable(sb.cuda()).squeeze(-1)
            la = la.cuda()
            lb = lb.cuda()
            n_feats = Variable(n_feats.cuda())
            label = label.type(torch.FloatTensor).cuda().view(-1, 1)
            mask_a = get_mask(sa, la, is_cuda=True)
            mask_b = get_mask(sb, lb, is_cuda=True)

            with torch.no_grad():
                pred = model(sa, sb, mask_a, mask_b, la, lb, n_feats)
                trn_labels.append(label.cpu().numpy())
                trn_preds.append(pred.cpu().numpy())
            valid_loss += loss_fn(pred, label).item()*len(sa)
            #break
        print('round %d, epoch %d' % (round, epoch))
        print('training')
        trn_preds = np.concatenate(trn_preds)
        trn_labels = np.concatenate(trn_labels)
        print('train loss　%f, val loss %f ' % (train_loss/len(train_index), valid_loss/len(test_index)))

        #find_best_acc(trn_preds, trn_labels)

    pred_vector.append(trn_preds)
    print(len(pred_vector))
    torch.save(model.state_dict(), 'model_type/model_type_%s.pth'% (round))
    round += 1

pred_vector = np.concatenate(pred_vector, axis=0)
np.save('model_type/deep_match.npy', pred_vector)

# 0.09
#0.144

