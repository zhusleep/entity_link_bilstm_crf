import numpy as np
import json
from tqdm import  tqdm as tqdm


class DataManager(object):
    def __init__(self):
        self.ner_list = ['O','B','I','E','S']
        np.random.seed(2019)

    def BIEOS(self, m):
        if len(m) == 1:
            return ['S']
        elif len(m) == 2:
            return ['B', 'E']
        elif len(m) > 2:
            result = ['I'] * len(m)
            result[0] = 'B'
            result[-1] = 'E'
            return result
        else:
            raise Exception('mention为空')

    def parseData(self, file_name, valid_num):
        X_arr = []
        ner_arr = []
        with open(file_name, 'r') as f:
            for line in tqdm(f):
                s = json.loads(line)
                X_arr.append(s['text'])
                ner = np.array(['O'] * len(s['text']))
                mention_ner = s['mention_data']
                for m in mention_ner:
                    ner[int(m['offset']):int(m['offset']) + len(m['mention'])] = self.BIEOS(m['mention'])

                label = []
                for label_type in ner:
                    label.append(self.ner_list.index(label_type))
                ner_arr.append(label)
        valid_index = np.random.choice(len(X_arr), valid_num, replace=False)
        train_X = []
        valid_X = []
        train_ner = []
        valid_ner = []
        for i in range(len(X_arr)):
            if i not in  valid_index:
                train_X.append(X_arr[i])
                train_ner.append(ner_arr[i])
            else:
                valid_X.append(X_arr[i])
                valid_ner.append(ner_arr[i])
        return train_X,train_ner,valid_X,valid_ner


data_manager = DataManager()


def read_kb(filename):
    #kb_data = []
    kb_dict = []
    with open(filename, 'r') as f:
        for line in f:
            item = json.loads(line)
            kb_dict.append(item['subject'])
            for alia in item['alias']:
                kb_dict.append(alia)
    return set(kb_dict)