#-*-coding:utf-8-*-
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling import BertModel
from torch.nn import init

from torchcrf import CRF
from tIme_distributed import TimeDistributed
from utils import sequence_cross_entropy_with_logits

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
                               num_layers=1,
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
        res = padded_res[desorted_indices]
        return res


class GRUEncoder(nn.Module):
    def __init__(self,
                 embed_size=200,
                 encoder_size=64,
                 bidirectional=True
                 ):
        super(GRUEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=embed_size,
                               hidden_size=encoder_size,
                               bidirectional=bidirectional,
                               num_layers=1,
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
        res = padded_res[desorted_indices]
        return res


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


# mask attention
class Attention(nn.Module):
    def __init__(self, feature_dim, bias=False, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        # if bias:
        #     self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = x.size()[1]

        # step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class CharModel(nn.Module):
    def __init__(self,
                 vocab_size=10000,
                 embed_size=300,
                 encoder_size=64,
                 bidirectional=True,
                 init_embedding=None):
        super(CharModel, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size,
                                           embed_size,
                                           padding_idx=0)
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.encoder_size = encoder_size
        self.bidirectional=bidirectional
        if init_embedding is not None:
            self.char_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.GRU = GRUEncoder(embed_size=self.embed_size,
                              encoder_size=self.encoder_size,
                              bidirectional=self.bidirectional)

    def forward(self, inputs):
        max_len = len(inputs[0])
        batch_size = len(inputs)
        reshaped_inputs = self._reshape_tensor(inputs)
        length = torch.tensor([len(x) for x in reshaped_inputs]).cuda()
        reshaped_inputs = nn.utils.rnn.pad_sequence(reshaped_inputs, batch_first=True).type(torch.LongTensor)
        reshaped_inputs = reshaped_inputs.cuda()
        char_input = self.char_embedding(reshaped_inputs)

        state = self.GRU(char_input, length)
        # final layer
        state = state[:, -1, :]
        state = state.view(batch_size, max_len, -1)
        #
        # # Now get the output back into the right shape.
        # # (batch_size, time_steps, **output_size)
        # new_size = reshaped_outputs.size()[1:]
        # outputs = reshaped_outputs.contiguous().view(new_size)
        # X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        # X = self.char_embedding(X)
        return state

    @staticmethod
    def _reshape_tensor(input_tensor):
        temp = []
        for sen in input_tensor:
            for sub_word in sen:
                temp.append(sub_word)
        return temp


class SPOModel(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 char_vocab_size=10000,
                 word_embed_size=300,
                 char_embed_size=300,
                 init_embedding=None,
                 char_init_embedding=None,
                 bidirectional=True,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.5,
                 seq_dropout=0.1,
                 pos_embed_size=100,
                 pos_dim=10):
        super(SPOModel, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embed_size,
                                           padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)
        self.seq_dropout = seq_dropout
        self.embed_size = word_embed_size
        self.embed_size += pos_dim
        self.embed_size += 2*encoder_size
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.LSTM = LSTMEncoder(embed_size=self.embed_size,
                                   encoder_size=encoder_size,
                                   bidirectional=bidirectional)
        self.GRU = GRUEncoder(embed_size=self.embed_size,
                                encoder_size=encoder_size,
                                bidirectional=bidirectional)
        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.attention = SoftAttention()
        self.lstm_attention = Attention(encoder_size*2)
        self.gru_attention = Attention(encoder_size*2)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(2*encoder_size),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2*encoder_size, out_features=64),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(64+dim_num_feat),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64+dim_num_feat, out_features=50),
            nn.Sigmoid()
        )
        self.char_model =CharModel(embed_size=char_embed_size,
                                   vocab_size=char_vocab_size,
                                       encoder_size=encoder_size,
                                       init_embedding=char_init_embedding,
                                       bidirectional=bidirectional)
        self.apply(self._init_qa_weights)

    def forward(self, X, pos_tags, mask_X, length, num_feats, char_vocab):
        batch_size = X.size()[0]
        X = self.word_embedding(X)
        pos_X = self.pos_embedding(pos_tags)
        X_char_mebedding = self.char_model(char_vocab)
        X = torch.cat([X, pos_X, X_char_mebedding], dim=-1)
        #X = torch.squeeze(self.dropout1d(torch.unsqueeze(X, -1)), -1)
        X1 = self.LSTM(X, length)

        v = self.lstm_attention(X1, mask=mask_X)

        #X2 = self.GRU(X, length)
        #v2 = self.gru_attention(X2, mask=mask_X)

        # mask_X = mask_X.view(batch_size, X.size()[1], 1)
        # # avg should be masked
        # va_avg = (torch.sum(X1, dim=1)/torch.sum(mask_X)).view(batch_size, -1)
        # va_max = F.adaptive_max_pool1d(X1.transpose(1, 2), output_size=1).view(batch_size, -1)
        # # avg should be masked
        # #va_avg2 = (torch.sum(X2, dim=1) / torch.sum(mask_X)).view(batch_size, -1)
        # #va_max2 = F.adaptive_max_pool1d(X2.transpose(1, 2), output_size=1).view(batch_size, -1)

        #v = torch.cat([v, v2], dim=-1)

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


class SPONerModel(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 embed_size=300,
                 init_embedding=None,
                 bidirectional=True,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.5,
                 seq_dropout=0.1,
                 pos_embed_size=100,
                 pos_dim=10,
                 ner_num=50):
        super(SPONerModel, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size,
                                           embed_size,
                                           padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)

        self.seq_dropout = seq_dropout
        self.embed_size = embed_size
        self.embed_size += pos_dim
        self.encoder_size = encoder_size
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.LSTM = LSTMEncoder(embed_size=self.embed_size,
                                   encoder_size=encoder_size,
                                   bidirectional=bidirectional)
        self.GRU = GRUEncoder(embed_size=self.embed_size,
                                encoder_size=encoder_size,
                                bidirectional=bidirectional)
        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.attention = SoftAttention()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(4*encoder_size),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4*encoder_size, out_features=64),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(64+dim_num_feat),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64+dim_num_feat, out_features=50),
            nn.Sigmoid()
        )
        self.NER = nn.Linear(4 * self.encoder_size, ner_num)
        #self.apply(self._init_qa_weights)


    def forward(self, X, pos_tags, mask_X, length, num_feats):
        batch_size = X.size()[0]
        X = self.word_embedding(X)
        pos_X = self.pos_embedding(pos_tags)
        X = torch.cat([X, pos_X], dim=-1)
        #X = torch.squeeze(self.dropout1d(torch.unsqueeze(X, -1)), -1)
        X1 = self.LSTM(X, length)
        va_max = F.adaptive_max_pool1d(X1.transpose(1, 2), output_size=1).view(batch_size, 1, -1)
        va_max_expand = va_max.expand(batch_size, X1.size()[1], 2*self.encoder_size)
        X2 = torch.cat([X1, va_max_expand], dim=2)  # (batch_size, seq_len, 256*2)

        logits = self.NER(X2)
        class_probabilities = F.softmax(logits, dim=2)

        #X3 = self.GRU(X, length)

        mask_X = mask_X.view(batch_size, X.size()[1], 1)
        # avg should be masked
        va_avg = (torch.sum(X1, dim=1)/torch.sum(mask_X)).view(batch_size, -1)
        va_max = F.adaptive_max_pool1d(X1.transpose(1, 2), output_size=1).view(batch_size, -1)
        # avg should be masked
        #va_avg2 = (torch.sum(X3, dim=1) / torch.sum(mask_X)).view(batch_size, -1)
        #va_max2 = F.adaptive_max_pool1d(X3.transpose(1, 2), output_size=1).view(batch_size, -1)

        v = torch.cat([va_avg, va_max], dim=-1)
        v = self.mlp(v)
        v = torch.cat([v, num_feats], dim=-1)
        v = self.mlp2(v)
        return logits, class_probabilities, v

        # return v


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        init.orthogonal_(self.weight)


class Linears(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hiddens,
                 bias=True,
                 activation='tanh'):
        super(Linears, self).__init__()
        assert len(hiddens) > 0

        self.in_features = in_features
        self.out_features = self.output_size = out_features

        in_dims = [in_features] + hiddens[:-1]
        self.linears = nn.ModuleList([Linear(in_dim, out_dim, bias=bias)
                                      for in_dim, out_dim
                                      in zip(in_dims, hiddens)])
        self.output_linear = Linear(hiddens[-1], out_features, bias=bias)
        self.activation = activation

    def forward(self, inputs):
        linear_outputs = inputs
        for linear in self.linears:
            linear_outputs = linear.forward(linear_outputs)
            if self.activation == 'tanh':
                linear_outputs = torch.tanh(linear_outputs)
            else:
                linear_outputs = torch.relu(linear_outputs)
        return self.output_linear.forward(linear_outputs)


class SPO_Model_Simple(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 word_embed_size=300,
                 init_embedding=None,
                 bidirectional=True,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.5,
                 seq_dropout=0.1,
                 num_tags=5
                ):
        super(SPO_Model_Simple, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embed_size,
                                           padding_idx=0)
        self.seq_dropout = seq_dropout
        self.embed_size = word_embed_size
        self.encoder_size = encoder_size
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.LSTM = LSTMEncoder(embed_size=self.embed_size,
                                   encoder_size=encoder_size,
                                   bidirectional=bidirectional)
        self.GRU = GRUEncoder(embed_size=self.embed_size,
                                encoder_size=encoder_size,
                                bidirectional=bidirectional)
        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.attention = SoftAttention()
        self.lstm_attention = Attention(encoder_size*2)
        self.gru_attention = Attention(encoder_size*2)
        self.crf_model = CRF(num_tags=num_tags, batch_first=True)
        self.apply(self._init_qa_weights)
        hidden_size = 100
        self.hidden = nn.Linear(2*self.encoder_size, hidden_size)
        self.NER = nn.Linear(hidden_size, num_tags)
        self.use_crf = True

    def cal_loss(self, X, mask_X, length, label=None):
        X = self.word_embedding(X)
        # X = torch.squeeze(self.dropout1d(torch.unsqueeze(X, -1)), -1)
        X1 = self.LSTM(X, length)
        X1 = self.hidden(X1)
        logits = self.NER(X1)
        if not self.use_crf:
            class_probabilities = F.softmax(logits, dim=2)
            loss = sequence_cross_entropy_with_logits(class_probabilities, label, weights=mask_X,
                                                      label_smoothing=False)
        else:
            loss = -1*self.crf_model(logits, label, mask=mask_X)
        return loss

    def forward(self, X, mask_X, length):
        # batch_size = X.size()[0]
        # seq_len = X.size()[1]
        X = self.word_embedding(X)
        X = torch.squeeze(self.dropout1d(torch.unsqueeze(X, -1)), -1)
        X1 = self.LSTM(X, length)
        X1 = self.hidden(X1)
        logits = self.NER(X1)
        if self.use_crf:
            pred = self.crf_model.decode(logits, mask=mask_X)
        else:
            pred = logits.argmax(dim=-1).cpu().numpy()
        return pred

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


class SPO_Model_Bert(nn.Module):
    def __init__(self,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.5,
                 seq_dropout=0.1,
                 num_tags=5
              ):
        super(SPO_Model_Bert, self).__init__()
        # self.word_embedding = nn.Embedding(vocab_size,
        #                                    word_embed_size,
        #                                    padding_idx=0)
        # self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)
        self.seq_dropout = seq_dropout

        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        #self.attention = SoftAttention()
        #self.lstm_attention = Attention(768)
        #self.gru_attention = Attention(encoder_size*2)
        #
        # self.mlp = nn.Sequential(
        #     nn.BatchNorm1d(768),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(in_features=768, out_features=64),
        #     nn.Sigmoid()
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.BatchNorm1d(64+dim_num_feat),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(in_features=64+dim_num_feat, out_features=50),
        #     nn.Sigmoid()
        # )
        bert_model = 'bert-base-chinese'
        self.bert = BertModel.from_pretrained(bert_model)
        self.use_layer = -1
        self.LSTM = LSTMEncoder(embed_size=768,
                                encoder_size=encoder_size,
                                bidirectional=True)
        hidden_size=100
        self.hidden = nn.Linear(2*encoder_size, hidden_size)
        self.NER = nn.Linear(hidden_size, num_tags)
        self.crf_model = CRF(num_tags=num_tags, batch_first=True)

        self.use_crf = True

    def cal_loss(self,token_tensor,mask_X,length,label=None):
        # self.bert.eval()
        # with torch.no_grad():
        bert_outputs, _ = self.bert(token_tensor, attention_mask=(token_tensor > 0).long(), token_type_ids=None,
                            output_all_encoded_layers=True)

        bert_outputs = torch.cat(bert_outputs[self.use_layer:], dim=-1)
        X1 = self.LSTM(bert_outputs, length)
        X1 = self.hidden(X1)
        logits = self.NER(X1)
        if not self.use_crf:
            class_probabilities = F.softmax(logits, dim=2)
            loss = sequence_cross_entropy_with_logits(class_probabilities, label, weights=mask_X,
                                                      label_smoothing=False)
        else:
            loss = -1 * self.crf_model(logits, label, mask=mask_X)
        return loss

    def forward(self, token_tensor, mask_X, length):
        batch_size = token_tensor.size()[0]

        self.bert.eval()
        with torch.no_grad():
            bert_outputs, _ = self.bert(token_tensor, attention_mask=(token_tensor > 0).long(), token_type_ids=None,
                                        output_all_encoded_layers=True)

        bert_outputs = torch.cat(bert_outputs[self.use_layer:], dim=-1)

        X1 = self.LSTM(bert_outputs, length)
        X1 = self.hidden(X1)
        logits = self.NER(X1)
        if self.use_crf:
            pred = self.crf_model.decode(logits, mask=mask_X)
        else:
            pred = logits.argmax(dim=-1).cpu().numpy()
        return pred


from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor


class EntityLink_bert(nn.Module):
    def __init__(self,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.2,
                 seq_dropout=0.1,
                 num_outputs=5
              ):
        super(EntityLink_bert, self).__init__()
        # self.word_embedding = nn.Embedding(vocab_size,
        #                                    word_embed_size,
        #                                    padding_idx=0)
        # self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)
        self.seq_dropout = seq_dropout

        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.span_extractor = EndpointSpanExtractor(encoder_size * 2)

        bert_model = 'bert-base-chinese'
        self.bert = BertModel.from_pretrained(bert_model)
        self.use_layer = -1
        self.LSTM = LSTMEncoder(embed_size=768,
                                encoder_size=encoder_size,
                                bidirectional=True)
        hidden_size = 100
        self.hidden = nn.Linear(2*encoder_size, hidden_size)
        self.classify = nn.Sequential(
            nn.BatchNorm1d(encoder_size*4),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=encoder_size*4, out_features=num_outputs)
        )

    def forward(self, token_tensor, mask_X, pos, length):
        self.bert.eval()
        with torch.no_grad():
            bert_outputs, _ = self.bert(token_tensor, attention_mask=(token_tensor > 0).long(),
                                        token_type_ids=None,
                                        output_all_encoded_layers=True)

        bert_outputs = torch.cat(bert_outputs[self.use_layer:], dim=-1)
        X1 = self.LSTM(bert_outputs, length)
        spans_contexts = self.span_extractor(
            X1,
            pos
        )
        pred = self.classify(spans_contexts.squeeze(0))
        #print(pred.size())
        return pred


class EntityLink(nn.Module):
    def __init__(self,
                 vocab_size,
                 init_embedding,
                 word_embed_size=300,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.2,
                 seq_dropout=0.1,
                 num_outputs=5
              ):
        super(EntityLink, self).__init__()
        # self.word_embedding = nn.Embedding(vocab_size,
        #                                    word_embed_size,
        #                                    padding_idx=0)
        # self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embed_size,
                                           padding_idx=0)
        self.seq_dropout = seq_dropout
        self.embed_size = word_embed_size
        self.encoder_size = encoder_size
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.seq_dropout = seq_dropout
        #self.lstm_attention = Attention(encoder_size*2)

        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.span_extractor = EndpointSpanExtractor(2*encoder_size)

        bert_model = 'bert-base-chinese'
        self.use_layer = -1
        self.LSTM = LSTMEncoder(embed_size=word_embed_size,
                                encoder_size=encoder_size,
                                bidirectional=True)
        hidden_size=100
        self.hidden = nn.Linear(2*encoder_size, hidden_size)
        self.classify = nn.Sequential(
            nn.BatchNorm1d(encoder_size*4),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=encoder_size*4, out_features=num_outputs)
        )

    def forward(self,token_tensor,mask_X,pos,length):
        X = self.word_embedding(token_tensor)

        X1 = self.LSTM(X, length)

        spans_contexts = self.span_extractor(
            X1,
            pos
        )
        #X2 = self.lstm_attention(X1)
        #X3 = torch.cat([spans_contexts.squeeze(0),X2],dim=-1)
        pred = self.classify(spans_contexts.squeeze(0))
        #print(pred.size())
        return pred


class EntityLink_entity_vector(nn.Module):
    def __init__(self,
                 vocab_size,
                 init_embedding,
                 word_embed_size=300,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.2,
                 seq_dropout=0.1,
                 num_outputs=5
              ):
        super(EntityLink_entity_vector, self).__init__()
        # self.word_embedding = nn.Embedding(vocab_size,
        #                                    word_embed_size,
        #                                    padding_idx=0)
        # self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embed_size,
                                           padding_idx=0)
        self.seq_dropout = seq_dropout
        self.embed_size = word_embed_size
        self.encoder_size = encoder_size
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.seq_dropout = seq_dropout
        #self.lstm_attention = Attention(encoder_size*2)

        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.span_extractor = EndpointSpanExtractor(2*encoder_size)

        bert_model = 'bert-base-chinese'
        self.use_layer = -1
        self.LSTM = LSTMEncoder(embed_size=word_embed_size,
                                encoder_size=encoder_size,
                                bidirectional=True)
        hidden_size=100
        self.hidden = nn.Linear(4*encoder_size, 100)
        self.classify = nn.Sequential(
            nn.BatchNorm1d(encoder_size*4),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=encoder_size*4, out_features=num_outputs)
        )
        self.nonlinear = nn.Tanh()

        self.hidden2tag = nn.Sequential(
            # nn.Linear(config.hidden_size * 2 + config.words_dim, config.hidden_size * 2),
            #nn.BatchNorm1d(6 * encoder_size),
            nn.Linear(6*encoder_size, 2*encoder_size),

            self.nonlinear,
            nn.Dropout(p=dropout),
            nn.Linear(2*encoder_size, 100)
        )

    def forward(self,token_tensor,mask_X,pos,vector,length):
        X = self.word_embedding(token_tensor)

        X1 = self.LSTM(X, length)

        spans_contexts = self.span_extractor(
            X1,
            pos
        )
        #self.hidden(spans_contexts)
        pred = self.hidden(spans_contexts)

        # X2 = torch.cat([X1, spans_contexts], dim=-1)
        # X2 = X2.permute(1,0,2)
        # pred = self.hidden2tag(X2)
        # pred = pred.permute(1,0,2)
        #grades = torch.sum(pred*vector)

        #X2 = self.lstm_attention(X1)
        #X3 = torch.cat([spans_contexts.squeeze(0),X2],dim=-1)
        #pred = self.classify(spans_contexts.squeeze(0))
        #print(pred.size())
        return pred


class EntityLink_v2(nn.Module):
    def __init__(self,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.2,
                 seq_dropout=0.1,
                 num_outputs=5
              ):
        super(EntityLink_v2, self).__init__()
        # self.word_embedding = nn.Embedding(vocab_size,
        #                                    word_embed_size,
        #                                    padding_idx=0)
        # self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)
        self.seq_dropout = seq_dropout

        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.span_extractor = EndpointSpanExtractor(768)

        bert_model = 'bert-base-chinese'
        self.bert = BertModel.from_pretrained(bert_model)
        self.use_layer = -1
        self.lstm_attention = Attention(768)

        self.LSTM = LSTMEncoder(embed_size=768,
                                encoder_size=encoder_size,
                                bidirectional=True)
        hidden_size = 100
        self.hidden = nn.Linear(2*encoder_size, hidden_size)
        self.span_linear = nn.Linear(768*2, 768)

        self.classify = nn.Sequential(
            nn.BatchNorm1d(encoder_size*4),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=encoder_size*4, out_features=num_outputs)
        )
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(4608),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4608, out_features=128),
            nn.ReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(128 + 2),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128+2, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, query, l_query, pos, candidate_abstract, l_abstract,
                candidate_labels, l_labels, candidate_type, candidate_abstract_numwords,
                candidate_numattrs, mask_abstract, mask_query, mask_labels):
        self.bert.eval()
        with torch.no_grad():
            query_bert_outputs, _ = self.bert(query, attention_mask=(query > 0).long(),
                                              token_type_ids=None,
                                              output_all_encoded_layers=True)
            query_bert_outputs = torch.cat(query_bert_outputs[self.use_layer:], dim=-1)
            candidate_abstract_bert_outputs, _ = self.bert(candidate_abstract, attention_mask=(candidate_abstract > 0).long(),
                                                           token_type_ids=None,
                                                           output_all_encoded_layers=True)
            candidate_abstract_bert_outputs = torch.cat(candidate_abstract_bert_outputs[self.use_layer:], dim=-1)
            candidate_labels_bert_outputs, _ = self.bert(candidate_labels, attention_mask=(candidate_labels > 0).long(),
                                                           token_type_ids=None,
                                                           output_all_encoded_layers=True)
            candidate_labels_bert_outputs = torch.cat(candidate_labels_bert_outputs[self.use_layer:], dim=-1)

        #query_lstm = self.LSTM(query_bert_outputs, l_query)
        query_attn_pool = self.lstm_attention(query_bert_outputs, mask=mask_query)

        #candidate_abstract_lstm = self.LSTM(candidate_abstract_bert_outputs, l_abstract)
        candidate_abstract_pool = self.lstm_attention(candidate_abstract_bert_outputs, mask=mask_abstract)

        #candidate_labels_lstm = self.LSTM(candidate_labels_bert_outputs, l_labels)
        candidate_labels_pool = self.lstm_attention(candidate_labels_bert_outputs, mask=mask_labels)

        spans_contexts = self.span_extractor(query_bert_outputs, pos)
        spans_contexts = self.span_linear(spans_contexts)

        relation_vector = torch.cat([query_attn_pool, candidate_abstract_pool, candidate_labels_pool,
                   query_attn_pool*candidate_abstract_pool,
                   candidate_labels_pool*query_attn_pool,
                   spans_contexts*candidate_labels_pool,
                   ], dim=-1)
        # print(relation_vector.size())

        relation_vector = self.mlp(relation_vector)
        relation_vector = torch.cat([relation_vector,
                                     candidate_abstract_numwords.unsqueeze(1),
                                     candidate_numattrs.unsqueeze(1)], dim=-1)
        # print(relation_vector.size())
        output = self.mlp2(relation_vector)
        # pred = self.classify(spans_contexts.squeeze(0))

        return output


class EntityLink_v3(nn.Module):
    def __init__(self,
                 vocab_size,
                 init_embedding,
                 word_embed_size=300,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.2,
                 seq_dropout=0.1,
                 num_outputs=5
                 ):
        super(EntityLink_v3, self).__init__()
        # self.word_embedding = nn.Embedding(vocab_size,
        #                                    word_embed_size,
        #                                    padding_idx=0)
        # self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embed_size,
                                           padding_idx=0)
        self.seq_dropout = seq_dropout
        self.embed_size = word_embed_size
        self.encoder_size = encoder_size
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        for param in self.word_embedding.parameters():
                param.requires_grad = False

        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.span_extractor = EndpointSpanExtractor(encoder_size*2)

        bert_model = 'bert-base-chinese'
        #self.bert = BertModel.from_pretrained(bert_model)
        self.use_layer = -1
        self.query_attention = Attention(encoder_size*2)
        self.abstract_attention = Attention(encoder_size*2)
        self.lstm_attention = Attention(encoder_size*2)


        self.LSTM_query = LSTMEncoder(embed_size=300,
                                encoder_size=encoder_size,
                                bidirectional=True)
        self.LSTM_abstract = LSTMEncoder(embed_size=300,
                                encoder_size=encoder_size,
                                bidirectional=True)
        self.LSTM = LSTMEncoder(embed_size=300,
                                encoder_size=encoder_size,
                                bidirectional=True)
        hidden_size = 100
        self.hidden = nn.Linear(2*encoder_size, hidden_size)
        self.span_linear = nn.Linear(encoder_size*4, encoder_size*2)

        self.classify = nn.Sequential(
            nn.BatchNorm1d(encoder_size*4),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=encoder_size*4, out_features=num_outputs)
        )
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=768, out_features=128),
            nn.ReLU(inplace=True)

        )
        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(128 + 2),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128+2, out_features=1),
            nn.Sigmoid()
        )
        # chanel_num = 1
        # filter_sizes = [1,2,3,5]
        # filter_num = 50
        # self.convs = nn.ModuleList(
        # [nn.Conv2d(chanel_num, filter_num, (size, word_embed_size)) for size in filter_sizes])

    def forward(self, query, l_query, pos, candidate_abstract, l_abstract,
                candidate_labels, l_labels, candidate_type, candidate_abstract_numwords,
                candidate_numattrs, mask_abstract, mask_query, mask_labels):
        query_bert_outputs = self.word_embedding(query)
        candidate_abstract_bert_outputs = self.word_embedding(candidate_abstract)
        candidate_labels_bert_outputs = self.word_embedding(candidate_labels)

        query_lstm = self.LSTM_query(query_bert_outputs, l_query)
        query_attn_pool = self.query_attention(query_lstm, mask=mask_query)

        candidate_abstract_lstm = self.LSTM_abstract(candidate_abstract_bert_outputs, l_abstract)
        candidate_abstract_pool = self.abstract_attention(candidate_abstract_lstm, mask=mask_abstract)

        # candidate_labels_lstm = self.LSTM(candidate_labels_bert_outputs, l_labels)
        # candidate_labels_pool = self.lstm_attention(candidate_labels_lstm, mask=mask_labels)

        # spans_contexts = self.span_extractor(query_lstm, pos)
        # spans_contexts = self.span_linear(spans_contexts)

        relation_vector = torch.cat([query_attn_pool, candidate_abstract_pool,
                   query_attn_pool*candidate_abstract_pool
                   # candidate_labels_pool*query_attn_pool,
                   # spans_contexts*candidate_labels_pool,
                   ], dim=-1)
        # print(relation_vector.size())

        relation_vector = self.mlp(relation_vector)
        relation_vector = torch.cat([relation_vector,
                                     candidate_abstract_numwords.unsqueeze(1),
                                     candidate_numattrs.unsqueeze(1)], dim=-1)
        # print(relation_vector.size())

        output = self.mlp2(relation_vector)
        # pred = self.classify(spans_contexts.squeeze(0))

        return output


# Mask attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        #print(hidden.shape, encoder_output.shape)
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, mask):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs, mask)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs, mask)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs, mask)
        #print(attn_energies.shape)
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        #print(F.softmax(attn_energies, dim=1).shape)
        #print(F.softmax(attn_energies, dim=1).unsqueeze(1).shape)
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
