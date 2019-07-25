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
from sklearn.externals import joblib


file_namne = 'data/raw_data/train.json'
data_all = data_manager.read_deep_match_test('data/final_test.pkl')
seed_torch(2019)

t = joblib.load('data/deep_match_tokenizer.pkl')

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
train_batch_size = 8000
valid_dataset = QAPair(data_all, t, data_manager.predictor)
valid_dataloader = DataLoader(valid_dataset, collate_fn=qapair_collate_fn, shuffle=False, batch_size=train_batch_size)
for r in tqdm(range(5)):

    model = QAModel(vocab_size=embedding_matrix.shape[0], embed_size=300, encoder_size=64, dropout=0.5,
                    seq_dropout=0.2, init_embedding=embedding_matrix, dim_num_feat=346)
    model.load_state_dict(torch.load('model_type/model_type_%d_3.pth' % r))

    use_cuda = True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)

    model.eval()
    valid_loss = 0
    trn_labels, trn_preds = [], []
    for sa, sb, la, lb, numerical_features, label in tqdm(valid_dataloader):
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
            trn_preds.append(pred.cpu().numpy())
            # break
    trn_preds = np.concatenate(trn_preds)

# torch.save(model.state_dict(), 'model_type/model_type_%d_%d.pth' % (round, epoch))

    pred_vector.append(trn_preds)

print(len(pred_vector))

pred_vector = np.concatenate(pred_vector, axis=0)
np.save('model_type/deep_match_test.npy', pred_vector)

# 0.09
#0.144

