import numpy as np
import json
from tqdm import tqdm as tqdm
import pickle, os, re
import pandas as pd
from gensim.models import Word2Vec


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
        self.read_basic_info()

        X_arr = []
        ner_arr = []
        id_arr = []
        wrong_label = 0
        with open(file_name, 'r') as f:
            for line in tqdm(f):
                s = eval(str(json.loads(line)).lower())
                X_arr.append(s['text'])
                ner = np.array(['O'] * len(s['text']))
                mention_ner = s['mention_data']
                id_arr.append(s['text_id'])
                # 过滤掉错误的标注
                temp_zero = np.zeros(len(s['text']))
                wrong = False
                for m in mention_ner:
                    #temp.append([m['offset'],m['offset']+len(m['mention']),m['mention']])
                    for offset in range(int(m['offset']), int(m['offset'])+len(m['mention'])):
                        if temp_zero[offset]==0:
                            temp_zero[offset] = 1
                        else:
                            wrong = True
                            break
                if wrong:
                    print(mention_ner)
                    wrong_label += 1
                    # 长句优先，短句去掉
                    temp = []
                    for m in mention_ner:
                        temp.append([int(m['offset']), int(m['offset'])+len(m['mention']), len(m['mention']), m['mention']])
                    temp = sorted(temp,key=lambda x:x[2],reverse=True)
                    temp_zero = np.zeros(len(s['text']))
                    new_mention_ner = []
                    for m in temp:
                        if np.sum(temp_zero[m[0]:m[1]]) != 0:
                            continue
                        else:
                            if m[3] not in self.name_id:
                                continue
                            temp_zero[m[0]:m[1]] = 1
                            new_mention_ner.append({'offset': m[0], 'mention': m[3]})

                    mention_ner = new_mention_ner
                #
                for m in mention_ner:
                    ner[int(m['offset']):int(m['offset']) + len(m['mention'])] = self.BIEOS(m['mention'])

                label = []
                for label_type in ner:
                    label.append(self.ner_list.index(label_type))
                ner_arr.append(label)
        print('错误标注样本数%d' % wrong_label)
        #valid_index = np.random.choice(len(X_arr), valid_num, replace=False)
        valid_index = np.arange(80000, 90001)
        train_X = []
        valid_X = []
        train_ner = []
        valid_ner = []
        train_id = []
        valid_id = []
        for i in range(len(X_arr)):
            if i not in valid_index:
                train_X.append(X_arr[i])
                train_ner.append(ner_arr[i])
                train_id.append(id_arr[i])
            else:
                valid_X.append(X_arr[i])
                valid_ner.append(ner_arr[i])
                valid_id.append(id_arr[i])
        return train_X,train_ner,train_id,valid_X,valid_ner,valid_id

    def parseData_predict(self,train_filename,test_filename):
        self.read_basic_info()
        X_arr = []
        ner_arr = []
        wrong_label = 0
        with open(train_filename, 'r') as f:
            for line in tqdm(f):
                s = eval(str(json.loads(line)).lower())
                X_arr.append(s['text'])
                ner = np.array(['O'] * len(s['text']))
                mention_ner = s['mention_data']
                # 过滤掉错误的标注
                temp_zero = np.zeros(len(s['text']))
                wrong = False
                for m in mention_ner:
                    # temp.append([m['offset'],m['offset']+len(m['mention']),m['mention']])
                    for offset in range(int(m['offset']), int(m['offset']) + len(m['mention'])):
                        if temp_zero[offset] == 0:
                            temp_zero[offset] = 1
                        else:
                            wrong = True
                            break
                if wrong:
                    print(mention_ner)
                    wrong_label += 1
                    # 长句优先，短句去掉
                    temp = []
                    for m in mention_ner:
                        temp.append(
                            [int(m['offset']), int(m['offset']) + len(m['mention']), len(m['mention']), m['mention']])
                    temp = sorted(temp, key=lambda x: x[2], reverse=True)
                    temp_zero = np.zeros(len(s['text']))
                    new_mention_ner = []
                    for m in temp:
                        if np.sum(temp_zero[m[0]:m[1]]) != 0:
                            continue
                        else:
                            if m[3] not in self.name_id:
                                continue
                            temp_zero[m[0]:m[1]] = 1
                            new_mention_ner.append({'offset': m[0], 'mention': m[3]})

                    mention_ner = new_mention_ner
                #
                for m in mention_ner:
                    ner[int(m['offset']):int(m['offset']) + len(m['mention'])] = self.BIEOS(m['mention'])

                label = []
                for label_type in ner:
                    label.append(self.ner_list.index(label_type))
                ner_arr.append(label)
        print('错误标注样本数%d' % wrong_label)

        X_arr_test = []
        with open(test_filename, 'r') as f:
            for line in tqdm(f):
                s = eval(str(json.loads(line)).lower())
                X_arr_test.append(s['text'])
        return X_arr, ner_arr, X_arr_test

    def parse_mention(self,file_name, valid_num):
        # type classification
        self.read_basic_info()

        kb_data = []
        kb = self.kb
        # kb_data.append(item)
        #---------------------读取数据库知识
        e_link = []
        for s in self.train_data:
            mention_ner = s['mention_data']
            for m in mention_ner:
                if m['kb_id'] == 'nil':
                    continue
                name_list = [kb[m['kb_id']]['subject']] + kb[m['kb_id']]['alias']
                if m['mention'] not in name_list:

                    # kb[m['kb_id']]['alias'] += [m['mention']]
                    keep = False
                    for name_c in ''.join(name_list):
                        if name_c in m['mention']:
                            # print(name_c,m['mention'])
                            keep = True
                            break

                    if not keep:
                        continue

                sentence = s['text']
                pos = [int(m['offset']), int(m['offset'])+len(m['mention'])-1]
                if pos[1] >= len(sentence):
                    raise Exception
                m_type = kb[m['kb_id']]['type'][0]
                e_link.append([s['text_id'], sentence, pos, m_type])

        type_list = self.type_list
        return e_link
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

    def deep_type_predict(self,filename):
        self.read_basic_info()
        test_data = self.read_test_data(filename)
        # type classification

        kb_data = []
        kb = self.kb
        # kb_data.append(item)
        #---------------------读取数据库知识
        e_link = []
        for s in test_data:
            mention_ner = s['mention_data']
            for m in mention_ner:
                if m['mention'] not in self.name_id: continue

                sentence = s['text']
                pos = [int(m['offset']), int(m['offset'])+len(m['mention'])-1]
                if pos[1] >= len(sentence):
                    raise Exception
                m_type = None
                e_link.append([s['text_id'], sentence, pos, m_type])

        type_list = self.type_list
        return e_link

    def parse_v2(self,file_name, valid_num):
        self.read_basic_info()
        data = pd.read_pickle('data/final.pkl').loc[0:10000, ['text_id','mention_start','train_mention','kb_id','label','num_candidates']]
        exclude = ['text_id',
                   'kb_id',
                   'train_mention',
                   'label',
                   'm_id', 'type']
        # exclude = []
        temp = []
        for x in data.columns:
            if x not in exclude: temp.append(x)
        self.predictor = temp
        #self.predictor = ['create_works']
        # self.predictor = ['label_mean',
        #  'label_count',
        #  'm_label_mean',
        #  'm_label_count']
        print(len(self.predictor))
        max_len=50
        data['question'] = data['text_id'].apply(lambda x: self.train_data[int(x)-1]['text'][0:max_len])

        def extract_info(kb):
            if not kb['data']:
                return '的'
            for item in kb['data']:
                if item['predicate']=='摘要':
                    first_sentence = item['object'].split('。')[0]
                    if first_sentence:
                        return first_sentence
            return '的'

        data['answer'] = data['kb_id'].apply(lambda x: extract_info(self.kb[x])[0:max_len])
        data['la'] = data['question'].apply(lambda x: len(x))
        data['lb'] = data['answer'].apply(lambda x: len(x))
        data.fillna(0,inplace=True)

        for item in self.predictor:
            data[item] = data[item].astype('float32')

        e_link = []

        for index, row in tqdm(data.iterrows()):
            e_link.append({'label': row['label'],
                           'query': row['question'],
                           'pos': [row['mention_start'],row['mention_start']+len(row['train_mention'])-1],
                           'candidate_abstract': row['answer'],
                           'candidate_type': self.type_list.index(self.kb[row['kb_id']]['type'][0]),
                           })
        return e_link

    def deep_distance(self):
        self.read_basic_info()
        data = pd.read_pickle('data/final.pkl').loc[:,
               ['text_id', 'mention_start', 'train_mention', 'kb_id', 'label', 'num_candidates']]
        exclude = ['text_id',
                   'kb_id',
                   'train_mention',
                   'label',
                   'm_id', 'type']
        # exclude = []
        temp = []
        for x in data.columns:
            if x not in exclude: temp.append(x)
        self.predictor = temp
        # self.predictor = ['create_works']
        # self.predictor = ['label_mean',
        #  'label_count',
        #  'm_label_mean',
        #  'm_label_count']
        print(len(self.predictor))
        max_len = 100
        data['question'] = data['text_id'].apply(lambda x: self.train_data[int(x) - 1]['text'][0:max_len])

        def extract_info(kb):
            if not kb['data']:
                return '的'
            name_list = [kb['subject']] +kb['alias']
            context = ''
            for item in kb['data']:
                context += item['object']
                context += item['predicate']

            return context

        data['answer'] = data['kb_id'].apply(lambda x: extract_info(self.kb[x])[0:max_len])
        data['la'] = data['question'].apply(lambda x: len(x))
        data['lb'] = data['answer'].apply(lambda x: len(x))
        data.fillna(0, inplace=True)

        for item in self.predictor:
            data[item] = data[item].astype('float32')

        e_link = []

        def find_ans_pos(kb,answer):
            name_list = [kb['subject']] +kb['alias']
            for n in name_list:
                if n in answer:
                    start = answer.index(n)
                    return [start, start+len(n)]
            return [0, 0]
        for index, row in tqdm(data.iterrows()):
            e_link.append({'label': row['label'],
                           'query': row['question'],
                           'pos': [row['mention_start'], row['mention_start'] + len(row['train_mention']) - 1],
                           'pos_answer': find_ans_pos(self.kb[row['kb_id']],row['answer']),
                           'candidate_abstract': row['answer'],
                           'candidate_type': self.type_list.index(self.kb[row['kb_id']]['type'][0]),
                           })
        return e_link
        # for s in tqdm(self.train_data):
        #     mention_ner = s['mention_data']
        #     m_list = []
        #     for m in mention_ner:
        #         m_list.append(m['mention'])
        #     for m in mention_ner:
        #
        #         if m['mention'] not in self.name_id:  # 不能注销，因为有些实体没有or m['kb_id']=='nil':
        #             continue
        #         if m['kb_id'] != 'nil':
        #             name_list = [self.kb[m['kb_id']]['subject']] + self.kb[m['kb_id']]['alias']
        #             if m['mention'] not in name_list:
        #                 continue
        #         # query 特征
        #         sentence = s['text']
        #         # mention 特征
        #         pos = [int(m['offset']), int(m['offset']) + len(m['mention']) - 1]
        #         # kb candidate特征
        #         candidate_ids = self.name_id[m['mention']]
        #         # 结构化知识 摘要，type, 标签，属性信息,属性个数，摘要字数
        #         for m_candidate_id in candidate_ids:
        #             candidate_numattrs = 0
        #             candidate_abstract_numwords = 0
        #             candidate_detail = self.kb[m_candidate_id]
        #             candi_text = ''
        #             for predicate in candidate_detail['data']:
        #                 candidate_numattrs += 1
        #                 candi_text += predicate['predicate']
        #                 candi_text += predicate['object']
        #
        #                 if predicate['predicate'] == '摘要':
        #                     candidate_abstract = predicate['object']
        #                     candidate_abstract_numwords = len(candidate_abstract)
        #             #                 if predicate['predicate'] == '标签':
        #             #                     candidate_label += predicate['object']
        #             if m_candidate_id == m['kb_id']:
        #                 label = 1
        #             else:
        #                 label = 0
        #             e_link.append({'label': label,
        #                            'query': sentence,
        #                            'pos': pos,
        #                            'candidate_abstract': candi_text,
        #                            'candidate_type': self.type_list.index(candidate_detail['type'][0]),
        #                            })
        # return e_link


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
        type_list_f = 'data/type_list.pkl'
        name_id_f = 'data/name_id.pkl'
        train_data_f = 'data/train_data.pkl'
        kb_data_f = 'data/kb_data.pkl'
        kb_f = 'data/kb.pkl'
        if os.path.exists('data/kb.pkl'):
            self.type_list = pickle.load(open(type_list_f,'rb'))
            self.name_id = pickle.load(open(name_id_f,'rb'))
            self.train_data = pickle.load(open(train_data_f,'rb'))
            self.kb_data = pickle.load(open(kb_data_f,'rb'))
            self.kb = pickle.load(open(kb_f,'rb'))
            return
        kb_data = []
        kb = {}
        type_list = []
        train_data = []
        with open('data/raw_data/train.json', 'r') as f:
            for line in f:
                raw = eval(str(json.loads(line)).lower())
                train_data.append(raw)
        with open('data/raw_data/kb_data', 'r') as f:
            for line in f:
                item = eval(str(json.loads(line)).lower())
                type_list.append(item['type'][0])
                kb[item['subject_id']] = item
                kb_data.append(item)

        type_list = list(set(type_list))
        self.type_list = type_list
        name_id = {}
        for kb_id in kb:
            for item in kb[kb_id]['alias']:
                if item not in name_id:
                    name_id[item] = [kb_id]
                else:
                    name_id[item].append(kb_id)
            if kb[kb_id]['subject'] not in name_id:
                name_id[kb[kb_id]['subject']] = [kb_id]
            else:
                name_id[kb[kb_id]['subject']].append(kb_id)
        for id in name_id:
            name_id[id] = sorted(list(set(name_id[id])))
        self.name_id = name_id
        self.train_data = train_data
        self.kb_data = kb_data
        self.kb = kb

        pickle.dump(self.type_list, open(type_list_f, 'wb'))
        pickle.dump(self.name_id, open(name_id_f, 'wb'))
        pickle.dump(self.train_data, open(train_data_f, 'wb'))
        pickle.dump(self.kb, open(kb_f, 'wb'))
        pickle.dump(self.kb_data, open(kb_data_f, 'wb'))

    def read_entity_embedding(self, file_name, train_num=200000):
        self.read_basic_info()
        entity_embedding = Word2Vec.load('embedding/w2v.model')
        e_link = []
        with open(file_name, 'r') as f:
            for line in tqdm(f):
                s = eval(str(json.loads(line)).lower())
                mention_ner = s['mention_data']
                for m in mention_ner:
                    if m['mention'] not in self.name_id:  # 不能注销，因为有些实体没有or m['kb_id']=='nil':
                        continue
                    if m['kb_id'] != 'nil':
                        name_list = [self.kb[m['kb_id']]['subject']] + self.kb[m['kb_id']]['alias']
                        if m['mention'] not in name_list:
                            continue
                    candidate_ids = self.name_id[m['mention']]
                    for m_candidate_id in candidate_ids:
                        if m_candidate_id==m['kb_id']:
                            label = 1
                        else:
                            label = -1
                        sentence = s['text']
                        # mention 特征
                        pos = [int(m['offset']), int(m['offset']) + len(m['mention']) - 1]
                        e_vector = entity_embedding.wv.word_vec(self.kb[m_candidate_id]['subject_id'])
                        e_link.append([sentence, pos, e_vector, label])

        train_num = train_num
        train_part = e_link[0:train_num]
        valid_part = e_link[train_num:]
        return train_part, valid_part

    def read_test_data(self,filename):
        # 戴安楠ner
        # ner_result = pd.read_csv(filename, sep='\t')
        # ner_result['offset'] = ner_result['offset'].apply(lambda x: eval(x.lower()))
        # ner_result['mention'] = ner_result['mention'].apply(lambda x: eval(x.lower()))
        # ner_result = ner_result.rename(columns={'mention': 'pred', 'offset': 'pos'})

        # my ner
        ner_result = pd.read_pickle('result/ner_bert_result.pkl')
        ner_result['text'] = ner_result['text'].apply(lambda x: ''.join(x[1:-1]))
        ner_result['pos'] = ner_result['pos'].apply(lambda x: [k[0] - 1 for k in x])
        ner_result['text_id'] = ner_result.index + 1
        test_data = []
        for index, row in ner_result.iterrows():
            item = {}
            item['text_id'] = row['text_id']
            item['text'] = row['text']
            item['mention_data'] = []
            for i, p in enumerate(row['pos']):
                if row['pred'][i] not in self.name_id:continue
                item['mention_data'].append({'mention': row['pred'][i], 'offset': str(p)})
            test_data.append(item)
        return test_data

    def read_deep_cosine_test(self, filename):
        self.read_basic_info()
        test_data = self.read_test_data(filename)

        entity_embedding = Word2Vec.load('embedding/w2v.model')
        e_link = []
        for s in tqdm(test_data):
            mention_ner = s['mention_data']
            for m in mention_ner:
                if m['mention'] not in self.name_id:  # 不能注销，因为有些实体没有or m['kb_id']=='nil':
                    continue
                candidate_ids = self.name_id[m['mention']]
                for m_candidate_id in candidate_ids:
                    label = -1
                    sentence = s['text']
                    # mention 特征
                    pos = [int(m['offset']), int(m['offset']) + len(m['mention']) - 1]
                    e_vector = entity_embedding.wv.word_vec(self.kb[m_candidate_id]['subject_id'])
                    e_link.append([sentence, pos, e_vector, label])
        return e_link



    def read_deep_match(self):
        self.read_basic_info()
        data = pd.read_pickle('data/final.pkl').iloc[:, :]
        exclude = ['text_id',
                   'kb_id',
                   'train_mention',
                   'label',
                   'm_id', 'type']
        # exclude = []
        temp = []
        for x in data.columns:
            if x not in exclude: temp.append(x)
        self.predictor = temp
        self.predictor = ['create_works']
        # self.predictor = ['label_mean',
        #  'label_count',
        #  'm_label_mean',
        #  'm_label_count']
        print(len(self.predictor))
        max_len=50
        data['question'] = data['text_id'].apply(lambda x: self.train_data[int(x)-1]['text'][0:max_len])

        def extract_info(kb):
            if not kb['data']:
                return '的'
            for item in kb['data']:
                if item['predicate']=='摘要':
                    first_sentence = item['object'].split('。')[0]
                    if first_sentence:
                        return first_sentence
            return '的'

        data['answer'] = data['kb_id'].apply(lambda x: extract_info(self.kb[x])[0:max_len])
        data['la'] = data['question'].apply(lambda x: len(x))
        data['lb'] = data['answer'].apply(lambda x: len(x))
        data.fillna(0,inplace=True)

        for item in self.predictor:
            data[item] = data[item].astype('float32')
        return data


    def data_enhance(self,max_len=50):
        # 读取基本信息
        self.read_basic_info()
        sentence_with_type = []
        for sen in tqdm(self.kb_data):
            name_list = [sen['subject']]+sen['alias']
            for m in sen['data']:
                if m['predicate'] == '摘要':
                    abstract = m['object']
                    abstract = abstract.split('。')
                    for sub_abstract in abstract:
                        for name in name_list:
                            sub_abstract = sub_abstract[0:max_len]
                            if name in sub_abstract:
                                left = sub_abstract.index(name)
                                right = left+len(name)-1
                                sentence_with_type.append([sub_abstract,left,right,
                                                           self.type_list.index(sen['type'][0]),
                                                           sen['subject_id']])

                        if len(sentence_with_type)<10:
                            print(sentence_with_type)
                        else:
                            continue

                else:
                    continue
        self.enhanced_data = sentence_with_type
        return sentence_with_type

data_manager = DataManager()


import ahocorasick


def read_kb(filename):
    A = ahocorasick.Automaton()
    train_data = []
    train_data_dict = {}
    with open('data/raw_data/train.json', 'r') as f:
        for line in f:
            raw = eval(str(json.loads(line)).lower())
            train_data.append(raw)
            train_data_dict[raw['text_id']] = raw
    kb = {}
    with open('data/raw_data/kb_data', 'r') as f:
        for line in f:
            item = eval(str(json.loads(line)).lower())
            kb[item['subject_id']] = item
    # # 补充别名实体进入kb
    # for s in train_data:
    #     for m in s['mention_data']:
    #         if m['kb_id'] == 'nil': continue
    #         if m['mention'] != kb[m['kb_id']]['subject'] and m['mention'] not in kb[m['kb_id']]['alias']:
    #             kb[m['kb_id']]['alias'] += [m['mention']]
    name_id = {}
    for kb_id in kb:
        for item in kb[kb_id]['alias']:
            if item not in name_id:
                name_id[item] = [kb_id]
            else:
                name_id[item].append(kb_id)
        if kb[kb_id]['subject'] not in name_id:
            name_id[kb[kb_id]['subject']] = [kb_id]
        else:
            name_id[kb[kb_id]['subject']].append(kb_id)
    for id in name_id:
        name_id[id] = list(set(name_id[id]))

    kb_dict = []
    for key,value in name_id.items():

        A.add_word(key, len(key))
    A.make_automaton()
    #return set(kb_dict)
    return A

