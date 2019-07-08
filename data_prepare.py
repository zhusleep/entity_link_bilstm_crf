import numpy as np
import json
from tqdm import  tqdm as tqdm
import pickle, os

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



    def parse_mention(self,file_name, valid_num):
        # type classification
        kb_data = []
        kb = {}
        with open('data/raw_data/kb_data', 'r') as f:
            for line in f:
                item = json.loads(line)
                kb[item['subject_id']] = item
                #kb_data.append(item)
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
                    pos = [int(m['offset']), int(m['offset'])+len(m['mention'])-1]
                    if pos[1]>=len(sentence):
                        raise Exception
                    m_type = kb[m['kb_id']]['type'][0]
                    type_list.append(m_type)
                    e_link.append([sentence, pos, m_type])
        print(c)
        type_list = list(set(type_list))
        self.type_list = type_list
        train_num = 200000
        train_part = e_link[0:200000]
        valid_part = e_link[200000:]

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





    def parse_v2(self,file_name, valid_num):
        if os.path.exists('data/features.pkl'):
            e_link = pickle.load(open('data/features.pkl', 'rb'))
            train_num = 2000
            train_part = e_link[0:train_num]
            valid_part = e_link[-500:]
            print('train size %d, valid size %d' % (len(train_part), len(valid_part)))
            return train_part, valid_part
        # type classification
        kb_data = []
        kb = {}
        type_list = []
        with open('data/raw_data/kb_data', 'r') as f:
            for line in f:
                item = json.loads(line)
                type_list.append(item['type'][0])
                kb[item['subject_id']] = item
                kb_data.append(item)
        type_list = list(set(type_list))
        self.type_list = type_list
        name_id = {}
        for kb_item in kb_data:
            for item in kb_item['alias']:
                if item not in name_id:
                    name_id[item] = [kb_item['subject_id']]
                else:
                    name_id[item].append(kb_item['subject_id'])
            if kb_item['subject'] not in name_id:
                name_id[kb_item['subject']] = [kb_item['subject_id']]
            else:
                name_id[kb_item['subject']].append(kb_item['subject_id'])
        # ---------------------读取数据库知识
        e_link = []

        with open(file_name, 'r') as f:
            for line in tqdm(f):
                s = json.loads(line)
                mention_ner = s['mention_data']
                for m in mention_ner:
                    if m['mention'] not in name_id:
                        continue
                    # query 特征
                    sentence = s['text']
                    # mention 特征
                    pos = [int(m['offset']), int(m['offset'])+len(m['mention'])-1]
                    # kb candidate特征
                    candidate_ids = name_id[m['mention']]
                    # 结构化知识 摘要，type, 标签，属性信息,属性个数，摘要字数
                    for m_candidate_id in candidate_ids:

                        candidate_abstract = ''
                        candidate_label = ''
                        candidate_abstract_numwords = 0  # 摘要的丰富程度
                        candidate_numattrs = 0  # 评价词条的丰富程度
                        candidate_detail = kb[m_candidate_id]
                        # todo query和abstract的重合程度/mention的tfidf特征
                        for predicate in candidate_detail['data']:
                            candidate_numattrs += 1
                            if predicate['predicate'] == '摘要':
                                candidate_abstract = predicate['object']
                                candidate_abstract_numwords = len(candidate_abstract)
                            if predicate['predicate'] == '标签':
                                candidate_label += predicate['object']
                        if m_candidate_id==m['kb_id']:
                            label = 1
                        else:
                            label = 0
                        e_link.append({'label': label,
                                       'query': sentence,
                                       'pos': pos,
                                       'candidate_abstract': candidate_abstract,
                                       'candidate_labels': candidate_label,
                                       'candidate_type': self.type_list.index(candidate_detail['type'][0]),
                                       'candidate_abstract_numwords': candidate_abstract_numwords,
                                       'candidate_numattrs': candidate_numattrs})

        train_num = 2000000
        train_part = e_link[0:train_num]
        valid_part = e_link[train_num:]
        pickle.dump(e_link, open('data/features.pkl', 'wb'))

        #valid_type = [type_list.index(x[2]) for x in valid_part]
        return train_part, valid_part

    def parse_v3(self,file_name, valid_num):
        if os.path.exists('data/features.pkl'):
            e_link = pickle.load(open('data/features.pkl', 'rb'))
            train_num = 5000
            train_part = e_link[0:train_num]
            valid_part = e_link[-2000:]
            # train_part = e_link[0:1000]
            # valid_part = e_link[200000:]
            print('train size %d, valid size %d' % (len(train_part), len(valid_part)))
            return train_part, valid_part
        # type classification
        kb_data = []
        kb = {}
        type_list = []
        with open('data/raw_data/kb_data', 'r') as f:
            for line in f:
                item = json.loads(line)
                type_list.append(item['type'][0])
                kb[item['subject_id']] = item
                kb_data.append(item)
        type_list = list(set(type_list))
        self.type_list = type_list
        name_id = {}
        for kb_item in kb_data:
            for item in kb_item['alias']:
                if item not in name_id:
                    name_id[item] = [kb_item['subject_id']]
                else:
                    name_id[item].append(kb_item['subject_id'])
            if kb_item['subject'] not in name_id:
                name_id[kb_item['subject']] = [kb_item['subject_id']]
            else:
                name_id[kb_item['subject']].append(kb_item['subject_id'])
        for id in name_id:
            name_id[id] = list(set(name_id[id]))
        # ---------------------读取数据库知识
        e_link = []

        with open(file_name, 'r') as f:
            for line in tqdm(f):
                s = json.loads(line)
                mention_ner = s['mention_data']
                for m in mention_ner:
                    if m['mention'] not in name_id:
                        continue
                    one_bacth = []
                    # query 特征
                    sentence = s['text']
                    # mention 特征
                    pos = [int(m['offset']), int(m['offset'])+len(m['mention'])-1]
                    # kb candidate特征
                    candidate_ids = name_id[m['mention']]
                    # 结构化知识 摘要，type, 标签，属性信息,属性个数，摘要字数
                    for m_candidate_id in candidate_ids:

                        candidate_abstract = ''
                        candidate_label = ''
                        candidate_abstract_numwords = 0  # 摘要的丰富程度
                        candidate_numattrs = 0  # 评价词条的丰富程度
                        candidate_detail = kb[m_candidate_id]
                        # todo query和abstract的重合程度/mention的tfidf特征
                        for predicate in candidate_detail['data']:
                            candidate_numattrs += 1
                            if predicate['predicate'] == '摘要':
                                candidate_abstract = predicate['object']
                                candidate_abstract_numwords = len(candidate_abstract)
                            if predicate['predicate'] == '标签':
                                candidate_label += predicate['object']
                        if m_candidate_id==m['kb_id']:
                            label = 1
                        else:
                            label = 0

                        one_bacth.append({'label': label,
                                       'query': sentence,
                                       'pos': pos,
                                       'candidate_abstract': candidate_abstract,
                                       'candidate_labels': candidate_label,
                                       'candidate_type': self.type_list.index(candidate_detail['type'][0]),
                                       'candidate_abstract_numwords': candidate_abstract_numwords,
                                       'candidate_numattrs': candidate_numattrs})
                    e_link.append(one_bacth)

        train_num = 80000
        train_part = e_link[0:train_num]
        valid_part = e_link[train_num:]
        pickle.dump(e_link, open('data/features.pkl', 'wb'))

        #valid_type = [type_list.index(x[2]) for x in valid_part]
        return train_part, valid_part

    def read_basic_info(self):
        kb_data = []
        kb = {}
        type_list = []
        with open('data/raw_data/kb_data', 'r') as f:
            for line in f:
                item = json.loads(line)
                type_list.append(item['type'][0])
                kb[item['subject_id']] = item
                kb_data.append(item)
        type_list = list(set(type_list))
        self.type_list = type_list
        name_id = {}
        for kb_item in kb_data:
            for item in kb_item['alias']:
                if item not in name_id:
                    name_id[item] = [kb_item['subject_id']]
                else:
                    name_id[item].append(kb_item['subject_id'])
            if kb_item['subject'] not in name_id:
                name_id[kb_item['subject']] = [kb_item['subject_id']]
            else:
                name_id[kb_item['subject']].append(kb_item['subject_id'])
        for id in name_id:
            name_id[id] = list(set(name_id[id]))
        self.name_id = name_id
        self.kb_data = kb_data
        self.kb = kb

    def read_entity_embedding(self, file_name):
        self.read_basic_info()
        e_link = []
        with open(file_name, 'r') as f:
            for line in tqdm(f):
                s = json.loads(line)
                mention_ner = s['mention_data']
                for m in mention_ner:
                    if m['mention'] not in self.name_id:
                        continue
                    one_bacth = []
                    # query 特征
                    sentence = s['text']
                    # mention 特征
                    pos = [int(m['offset']), int(m['offset']) + len(m['mention']) - 1]
                    # kb candidate特征
                    candidate_ids = self.name_id[m['mention']]
                    # 结构化知识 摘要，type, 标签，属性信息,属性个数，摘要字数
                    for m_candidate_id in candidate_ids:

                        candidate_abstract = ''
                        candidate_label = ''
                        candidate_abstract_numwords = 0  # 摘要的丰富程度
                        candidate_numattrs = 0  # 评价词条的丰富程度
                        candidate_detail = kb[m_candidate_id]
                        # todo query和abstract的重合程度/mention的tfidf特征
                        for predicate in candidate_detail['data']:
                            candidate_numattrs += 1
                            if predicate['predicate'] == '摘要':
                                candidate_abstract = predicate['object']
                                candidate_abstract_numwords = len(candidate_abstract)
                            if predicate['predicate'] == '标签':
                                candidate_label += predicate['object']
                        if m_candidate_id == m['kb_id']:
                            label = 1
                        else:
                            label = 0

                        one_bacth.append({'label': label,
                                          'query': sentence,
                                          'pos': pos,
                                          'candidate_abstract': candidate_abstract,
                                          'candidate_labels': candidate_label,
                                          'candidate_type': self.type_list.index(candidate_detail['type'][0]),
                                          'candidate_abstract_numwords': candidate_abstract_numwords,
                                          'candidate_numattrs': candidate_numattrs})
                    e_link.append(one_bacth)


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