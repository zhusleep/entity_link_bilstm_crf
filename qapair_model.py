#-*-coding:utf-8-*-
import torch
from torch import nn
from torch.nn import functional as F


class LSTMEncoder(nn.Module):
    def __init__(self,
                 embed_size=200,
                 encoder_size=64,
                 bidirectional=True
                 ):
        super(LSTMEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=encoder_size,
                               bidirectional=bidirectional,
                               batch_first=True)

    def forward(self, x, len_x):
        # 对输入的batch按照长度进行排序
        sorted_seq_lengths, indices = torch.sort(len_x, descending=True)
        # 排序前的顺序可通过再次排序恢复
        _, desorted_indices = torch.sort(indices, descending=False)
        x = x[indices]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(x, sorted_seq_lengths, batch_first=True)
        res, state = self.encoder(packed_inputs)
        padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)
        desorted_indices = padded_res[desorted_indices]
        return desorted_indices


def masked_softmax(x, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    if mask is None:
        result = F.softmax(x, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = F.softmax(x * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = x.masked_fill((1 - mask).byte(), mask_fill_value)
            result = F.softmax(masked_vector, dim=dim)

    return result


class SoftAttention(nn.Module):
    def forward(self, a, b, mask_a, mask_b):
        attention_matrix = a.bmm(b.transpose(2, 1).contiguous())
        attention_mask = mask_a.bmm(mask_b.transpose(2, 1).contiguous())

        weight_for_a = masked_softmax(attention_matrix, attention_mask, -1)
        attended_a = weight_for_a.bmm(b)

        attention_matrix = attention_matrix.transpose(2, 1).contiguous()
        attention_mask = attention_mask.transpose(2, 1).contiguous()
        weight_for_b = masked_softmax(attention_matrix, attention_mask, -1)
        attended_b = weight_for_b.bmm(a)

        return attended_a, attended_b


class QAModel_sim(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 embed_size=300,
                 init_embedding=None,
                 bidirectional=True,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.5,
                 seq_dropout=0.1):
        super(QAModel,self).__init__()
        self.word_embedding = nn.Embedding(vocab_size,
                                           embed_size,
                                           padding_idx=0)
        self.seq_dropout = seq_dropout
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.encoder = LSTMEncoder(embed_size=embed_size,
                                   encoder_size=encoder_size,
                                   bidirectional=bidirectional)
        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.attention = SoftAttention()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(2*4*4*encoder_size),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2*4*4*encoder_size, out_features=64),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(64+dim_num_feat),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64+dim_num_feat, out_features=1),
            nn.Sigmoid()
        )
        self.apply(self._init_qa_weights)

    def forward(self, sa, sb, mask_a, mask_b, la, lb, num_feats):
        batch_size = sa.size()[0]

        ea = self.word_embedding(sa)
        eb = self.word_embedding(sb)
        ea = torch.squeeze(self.dropout1d(torch.unsqueeze(ea, -1)), -1)
        eb = torch.squeeze(self.dropout1d(torch.unsqueeze(eb, -1)), -1)
        ea = self.encoder(ea, la)
        eb = self.encoder(eb, lb)
        mask_a = mask_a.view(batch_size, ea.size()[1], 1)
        mask_b = mask_b.view(batch_size, eb.size()[1], 1)

        aa, ab = self.attention(ea,eb,mask_a,mask_b)

        ma = torch.cat([ea, aa, ea-aa, ea*aa], dim=-1)
        mb = torch.cat([eb, ab, eb-ab, eb*ab], dim=-1)

        # avg should be masked
        va_avg = (torch.sum(ma, dim=1)/torch.sum(mask_a)).view(batch_size, -1)
        va_max = F.adaptive_max_pool1d(ma.transpose(1, 2), output_size=1).view(batch_size, -1)
        vb_max = F.adaptive_max_pool1d(mb.transpose(1, 2), output_size=1).view(batch_size, -1)
        vb_avg = (torch.sum(mb, dim=1)/torch.sum(mask_b)).view(batch_size, -1)

        v = torch.cat([va_avg, va_max, vb_avg, vb_max], dim=-1)
        v = self.mlp(v)
        v = torch.cat([v, num_feats], dim=-1)
        v = self.mlp2(v)

        return v

    @staticmethod
    def _init_qa_weights(module):
        '''
        Initialize the weights of the qa model
        :param module:
        :return:
        '''
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.LSTM):
            nn.init.xavier_normal_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0)
            nn.init.constant_(module.bias_hh_l0.data, 0)
            hidden_size = module.bias_hh_l0.data.shape[0]//4
            module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

            if module.bidirectional:
                nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
                nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
                module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0


class QAModel(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 embed_size=300,
                 init_embedding=None,
                 bidirectional=True,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.5,
                 seq_dropout=0.1):
        super(QAModel,self).__init__()
        self.word_embedding = nn.Embedding(vocab_size,
                                           embed_size,
                                           padding_idx=0)
        self.seq_dropout = seq_dropout
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.encoder = LSTMEncoder(embed_size=embed_size,
                                   encoder_size=encoder_size,
                                   bidirectional=bidirectional)
        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.attention = SoftAttention()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(2*4*4*encoder_size),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2*4*4*encoder_size, out_features=64),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(64+1),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64+1, out_features=1),
            nn.Sigmoid()
        )
        self.apply(self._init_qa_weights)

    def forward(self, sa, sb, mask_a, mask_b, la, lb, num_feats):
        batch_size = sa.size()[0]

        ea = self.word_embedding(sa)
        eb = self.word_embedding(sb)
        ea = torch.squeeze(self.dropout1d(torch.unsqueeze(ea, -1)), -1)
        eb = torch.squeeze(self.dropout1d(torch.unsqueeze(eb, -1)), -1)
        ea = self.encoder(ea, la)
        eb = self.encoder(eb, lb)
        mask_a = mask_a.view(batch_size, ea.size()[1], 1).float()
        mask_b = mask_b.view(batch_size, eb.size()[1], 1).float()

        aa,ab = self.attention(ea,eb,mask_a,mask_b)

        ma = torch.cat([ea, aa, ea-aa, ea*aa], dim=-1)
        mb = torch.cat([eb, ab, eb-ab, eb*ab], dim=-1)

        # avg should be masked
        va_avg = (torch.sum(ma, dim=1)/torch.sum(mask_a)).view(batch_size, -1)
        va_max = F.adaptive_max_pool1d(ma.transpose(1, 2), output_size=1).view(batch_size, -1)
        vb_max = F.adaptive_max_pool1d(mb.transpose(1, 2), output_size=1).view(batch_size, -1)
        vb_avg = (torch.sum(mb, dim=1)/torch.sum(mask_b)).view(batch_size, -1)

        v = torch.cat([va_avg, va_max, vb_avg, vb_max], dim=-1)
        v = self.mlp(v)
        v = torch.cat([v, num_feats], dim=-1)
        v = self.mlp2(v)

        return v

    @staticmethod
    def _init_qa_weights(module):
        '''
        Initialize the weights of the qa model
        :param module:
        :return:
        '''
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.LSTM):
            nn.init.xavier_normal_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0)
            nn.init.constant_(module.bias_hh_l0.data, 0)
            hidden_size = module.bias_hh_l0.data.shape[0]//4
            module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

            if module.bidirectional:
                nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
                nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
                module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0



