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
                    ner[int(m['offset']):int(m['offset']) + len(m['mention'])-1] = self.BIEOS(m['mention'])

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

    def parse_mention(self,file_name, valid_num):
        # type classification
        kb_data = []
        kb = {}
        with open('data/raw_data/kb_data', 'r') as f:
            for line in f:
                item = json.loads(line)
                kb[item['subject_id']] = item
                kb_data.append(item)
        #---------------------读取数据库知识
        e_link = []
        type_list = []
        c = 0
        with open(file_name, 'r') as f:
            for line in tqdm(f):
                s = json.loads(line)
                mention_ner = s['mention_data']
                for m in mention_ner:
                    if m['kb_id'] == 'NIL':
                        continue
                    sentence = s['text']
                    pos = [int(m['offset']), int(m['offset'])+len(m['mention'])]
                    m_type = kb[m['kb_id']]['type'][0]
                    type_list.append(m_type)
                    e_link.append([sentence, pos, m_type])
        print(c)
        type_list = list(set(type_list))
        self.type_list = type_list
        train_num = 200000
        train_part = e_link[0:train_num]
        valid_part = e_link[train_num:]
        train_X = [x[0] for x in train_part]
        train_pos = [x[1] for x in train_part]

        train_type = []
        for x in train_part:
            train_type.append(type_list.index(x[2]))

        #train_type = [type_list.index(x[2]) for x in train_part]
        valid_X = [x[0] for x in valid_part]
        valid_pos = [x[1] for x in valid_part]
        valid_type = []
        for x in valid_part:
            valid_type.append(type_list.index(x[2]))
        #valid_type = [type_list.index(x[2]) for x in valid_part]
        return train_X,train_pos,train_type,valid_X,valid_pos,valid_type


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