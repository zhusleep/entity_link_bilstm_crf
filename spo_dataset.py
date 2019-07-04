#-*-coding:utf-8-*-
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm as tqdm


class SPO(Dataset):
    def __init__(self, X, tokenizer, max_len=50, label=None,  ner=None, combined_char_t=None):
        super(SPO, self).__init__()
        self.max_len = max_len
        X = self.limit_length(X)
        self.raw_X = X
        self.label = label
        self.tokenizer = tokenizer
        self.X = tokenizer.transform(X)
        # X = pad_sequences(X, maxlen=198)
        self.length = [len(sen) for sen in self.X]
        # self.numerical_df = self.cal_numerical_f(self.X)
        self.ner = ner
        self.combined_char_t = combined_char_t

    def limit_length(self, X):
        temp = []
        for item in X:
            temp.append(item[0:self.max_len])
        return temp

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        ner = torch.tensor(self.ner[index])
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if  ner is not None:
            return index, sentence, ner,length
        else:
            return index, sentence,length


class SPO_BERT(Dataset):
    def __init__(self, X, tokenizer, label=None,  ner=None):
        super(SPO_BERT, self).__init__()
        self.raw_X = X
        self.label = label
        self.tokenizer = tokenizer
        self.X = self.deal_for_bert(X, self.tokenizer)
        # X = pad_sequences(X, maxlen=198)
        self.ner = ner
        self.length = [len(sen) for sen in self.X]

    def deal_for_bert(self,x,t):
        # text = {}
        # for s in x:
        #     for word in s:
        #         if word in text:
        #             text[word] += 1
        #         else:
        #             text[word] = 1
        # extra_token_id = 1
        # for w in sorted(text.items(), key=lambda x: x[1])[-90:]:
        #     self.tokenizer.vocab[w[0]] = extra_token_id
        #     extra_token_id += 1

        bert_tokens = []
        for item in x:
            temp = []
            for w in item:
                if w in self.tokenizer.vocab:
                    temp.append(w)
                else:
                    temp.append('[UNK]')

            #sen = t.tokenize(''.join(item))
            indexed_tokens = t.convert_tokens_to_ids(temp)
            bert_tokens.append(indexed_tokens)
        return bert_tokens

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        ner = torch.tensor(self.ner[index])
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if  ner is not None:
            return index, sentence, ner, length
        else:
            return index, sentence, length


class SPO_BERT_LINK(Dataset):
    def __init__(self, X, tokenizer, pos, type=None):
        super(SPO_BERT_LINK, self).__init__()
        self.raw_X = X
        self.type = type
        self.tokenizer = tokenizer
        self.X = self.deal_for_bert(X, self.tokenizer)
        # X = pad_sequences(X, maxlen=198)
        self.pos = pos
        self.length = [len(sen) for sen in self.X]

    def deal_for_bert(self,x,t):
        bert_tokens = []
        for item in x:
            temp = []
            for w in item:
                if w in self.tokenizer.vocab:
                    temp.append(w)
                else:
                    temp.append('[UNK]')

            #sen = t.tokenize(''.join(item))
            indexed_tokens = t.convert_tokens_to_ids(temp)
            bert_tokens.append(indexed_tokens)
        return bert_tokens

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        pos = self.pos[index]
        type = self.type[index]
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if  type is not None:
            return index, sentence, type, pos, length
        else:
            return index, sentence,length



def pad_sequence(sequences):
    max_len = max([len(s) for s in sequences])
    out_mat = [[torch.tensor([0])]*max_len]*len(sequences)
    for i, sen in enumerate(sequences):
        sen = [torch.tensor(w) for w in sen]
        length = len(sen)
        out_mat[i][:length] = sen

    return out_mat


def collate_fn_withchar(batch):

    if len(batch[0]) == 7:
        index, X, pos, length, numerical_feats, char_vocab, label = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        numerical_feats = torch.tensor(numerical_feats)
        label = torch.tensor(label)
        char_vocab = pad_sequence(char_vocab)
        return index, X, pos, length, numerical_feats, char_vocab, label
    else:
        index, X, pos, length, numerical_feats = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        numerical_feats = torch.tensor(numerical_feats)
        return index, X, pos, length, numerical_feats


def collate_fn(batch):

    if len(batch[0]) == 4:
        index, X, ner, length = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        return index, X, ner, length,
    else:
        index, X, length = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        return index, X, length,


def collate_fn_link(batch):

    if len(batch[0]) == 5:
        index, sentence, type, pos, length = zip(*batch)
        pos = torch.tensor(pos, dtype=torch.int)
        length = torch.tensor(length, dtype=torch.int)
        type= torch.tensor(type, dtype=torch.long)
        return index, sentence, type, pos, length,
    else:
        index, X, length = zip(*batch)
        length = torch.tensor(length, dtype=torch.long)
        return index, X, length,


def collate_fn_ner(batch):

    if len(batch[0]) == 7:
        index, X, pos, length, numerical_feats, ner, label = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        numerical_feats = torch.tensor(numerical_feats)
        label = torch.tensor(label)
        return index, X, pos, length, numerical_feats, ner, label
    else:
        index, X, pos, length, numerical_feats = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        numerical_feats = torch.tensor(numerical_feats)
        return index, X, pos, length, numerical_feats


def get_mask(sequences_batch, sequences_lengths, is_cuda=True):
    """
    Get the mask for a batch of padded variable length sequences.
    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).
    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.uint8)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    if is_cuda:
        return mask.cuda()
    else:
        return mask


