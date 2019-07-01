#-*-coding:utf-8-*-

#coding:utf-8

from sklearn.feature_extraction.text import CountVectorizer
from tokenize_pkg.langconv import *
import re
import jieba
import pandas as pd
import os

# 获得根目录
root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))


class Tokenizer(object):
    '''
    A custom tokenizer.
    '''

    def __init__(self, max_feature=5000, pad_word='<PAD>', min_df=1, lowercase=True,
                 token_pattern=r"(?u).+",
                 segment=True,
                 domain_word_path=None,
                 delword_path=None,
                 short2full_path=None,
                 synonym_path=None,
                 stop_words_path=None):
        self.voc = {}
        self.index2word = {}
        self.max_feature = max_feature
        self.pad_word = pad_word
        self.min_df = min_df
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.segment = segment
        self.num_words = 0
        # 加减字典
        self.domain_words = set()
        if domain_word_path is not None:                      # 增加文件路径判断增加使用灵活性,允许为空
            with open(domain_word_path, 'r', encoding="utf-8") as f:
                for l in f:
                    l = l.strip()
                    self.domain_words.add(l)
                    jieba.add_word(l, freq=10000, tag=None)
        if delword_path is not None:
            with open(delword_path, 'r', encoding="utf-8") as fd:
                for l in fd:
                    l = l.strip()
                    jieba.del_word(l)

        # 停用词表
        self.stop_words = []
        if stop_words_path is not None:
            with open(stop_words_path, 'r', encoding="utf-8") as fs:
                for l in fs:
                    l = l.strip()
                    self.stop_words.append(l)

        # 初始化
        #self.vectorizer = CountVectorizer(max_features=max_feature, token_pattern=r"(?u).+",stop_words=self.stop_words) #(?u)\b\w+\b

        # 缩写转全称的文件读入
        if short2full_path is not None:
            short2full_df = pd.read_excel(short2full_path, sheet_name='Short2Full')
            self.Short2Full_dict = short2full_df.set_index('原始词').T.to_dict('records')[0]
            for i in self.Short2Full_dict.keys():  # 英文缩写全部变大写
                self.Short2Full_dict[i.upper()] = self.Short2Full_dict.pop(i)

        # 同义词词表
        if synonym_path is not None:
            sy_q = pd.read_excel(synonym_path, sheet_name='同义词')
            self.synonym_dict = sy_q.set_index('原始词').T.to_dict('records')[0]
            for i in self.synonym_dict.keys():
                self.synonym_dict[re.sub('[,，]{1}', '|', i)] = self.synonym_dict.pop(i)

        # 替换规则
        self.ReplaceDict = {
            'Vehicle_identify_number': r'(?P<Vehicle_identify_number>lsfa[a-zA-Z0-9]*)',           # 车架号
            'Phone_number'           : r'(?P<Phone_number>(1[3-9]\d{9}))',                         # 手机号码
            'Order_number'           : r'(?P<Order_number>a*1001[0-9]*)',                          # 订单编号
            'Customer_service_number': r'(?P<Customer_service_number>4000812011)',                 # 客服电话
            'Email'                  : r'(?P<Email>[\w]+(\.[\w]+)*@[\w]+(\.com))',                 # 电子邮箱
            'URL'                    : r'(?P<URL>(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]'
                                       r'|(?:%[0-9a-fA-F][0-9a-fA-F]))+))',  # 网址
            'Taobao_password'        : r'(?P<Taobao_password>(￥{1}(\w)+￥{1}))',                   # 淘口令
            'ID_number'              : r'(?P<ID_number>([1-6]\d{5}[12]\d{3}(0[1-9]|1[12])(0[1-9]|'
                                       r'1[0-9]|2[0-9]|3[01])\d{3}(\d|X|x)))',  # 身份证
            'Fixed_telephone'        : r'(?P<Fixed_telephone>(\(?0\d{2,3}[)-]?\d{7,8}))',          # 固定电话
            'Date'                   : r'(?P<Date>([012][0-9]{3}[-/](([1][012])|([0]?[1-9]))[-/]'
                                       r'(([3][01])|([12][0-9])|([0]?[1-9]))))',
        # 日期
            'Time'                   : r'(?P<Time>(([2][0-4]|[1][0-9]|[0]?[0-9])([:])([1-5][0-9]|'
                                       r'[0]?[0-9])([:])([1-5][0-9]|[0]?[0-9])))',
        # 时间
            'Trouble_code'           : r'(?P<Trouble_code>(p\d{4}))'                                # 故障码
        }
        self.replace_word = set() # 用于替换的新词构建的词典
        for i in self.ReplaceDict.keys():
            self.replace_word.add(str('<' + i + '>'))

        # 单位替换字典
        self.Unit_dict = {'μm': '微米', 'nm': '纳米', 'mm': '毫米', 'cm': '厘米', 'dm': '分米', 'km': '千米',  # 长度单位  'm':'米'
                     'kg': '千克',  # 重量      'g':'克',
                     # 'w':'瓦','kw':'千瓦',                                                            # 功率
                     'kwh': '千瓦时',  # 能量
                     # 'l':'升',                                                                       # 体积单位
                     '°': '度',  # 角度单位
                     'kpa': '千帕', 'Mpa': '兆帕',  # 压强单位
                     'kV': '千伏',  # 电压单位    V': '伏',
                     # 'ka':'千安','a':'安培','mA':'毫安','μA':'微安','nA':'纳安',                        # 电流单位
                     # 's':'秒','min':'分钟','h':'小时',                                                # 时间单位 时间 h min s
                     }

    def fit(self, all_sentences):
        # 参考 skelarn.feature_extraction.text,和transform融合一起

        self.fit_transform(all_sentences)

    def fit_transform(self, all_sentences):
        # 输入：原始数据 all_sentences(列表)   输出：voc
        # all_words = []
        # for s in all_sentences:
        #     #x = ' '.join(jieba.lcut(self.preprocessing(s))).split()
        #     x =  jieba.lcut(self.preprocessing(s))
        #     for word in x:
        #         all_words.append(word)

        voc = {}
        new_sentence = []
        for line in all_sentences:
            if self.segment:

                line = jieba.lcut(self.preprocessing(line))

                line = map(str, line)
                # stopwords/lowercase/pattern 放在preprocessing中处理,未完成
                new_sentence.append(line)
                for each in line:
                    if each in voc:
                        voc[each] += 1
                    else:
                        voc[each] = 1
            else:
                line = self.preprocessing(line)
                line = map(str, line)
                # stopwords/lowercase/pattern 放在preprocessing中处理,未完成
                new_sentence.append(line)
                for each in line:
                    if each in voc:
                        voc[each] += 1
                    else:
                        voc[each] = 1

        # min_df
        for key, value in voc.items():
            if value < self.min_df:
                del voc[key]
        # max_feature
        voc_sort = sorted(voc.items(), key=lambda x: x[1], reverse=True)
        if len(voc_sort) > self.max_feature:
            drop_key = [x[0] for x in voc_sort[self.max_feature:]]
            for key in drop_key:
                del voc[key]
        self.word_num = voc
        # 排序生成
        voc_sort = sorted(voc.items(), key=lambda x: x[1], reverse=True)
        for i, item in enumerate(voc_sort):
            self.voc[item[0]] = i+10 # index  从10开始,在复杂任务中比如seq2seq会出现一些复杂token 比如 pad,sos,eos等

        # all_words = list(set(all_words))  # 这样写可以加速 all_words 不可以去重

        # self.vectorizer.fit(all_words)
        # self.voc = dict(self.vectorizer.vocabulary_)

        # self.voc[self.pad_word] = 0
        self.voc['UNK'] = 0
        # self.voc[self.PAD]

        for key, value in self.voc.items():
            self.index2word[value] = key

        print('total word lcut : %d' % len(self.voc))
        #       lcut_for_search 继续加入tokenizer
        #       for s in all_sentences:
        #           words = jieba.lcut_for_search(s.replace(' ', ''))
        #           for w in words:
        #               self.addword(w)
        # 存储tokenizer
        # 加入字典
        for w in self.domain_words:
            self.addword(w)
        for w in self.replace_word:
            self.addword(w)
        print('total word addword: %d' % len(self.voc))
        self.num_words = max(self.voc.values())

        return [[self.voc[i] if i in self.voc else self.voc['UNK'] for i in x] for x in new_sentence]

    def cut(self, x):
        x_ = ' '.join(jieba.lcut(self.preprocessing(x))).split()
        return x_

    def transform(self, x):
        new_sentence = []
        for line in x:
            if self.segment:
                line = jieba.lcut(self.preprocessing(line))
            else:
                line = self.preprocessing(line)
            new_sentence.append(line)

        if self.voc is None:
            raise ValueError('The tokenizer has not been trained yet!')
        return [[self.voc[i] if i in self.voc else self.voc['UNK'] for i in x] for x in new_sentence]

    def addSentence(self, sentence):
        if self.segment:
            for word in jieba.lcut(sentence):
                self.addWord(word)
        else:
            for word in sentence:
                self.addWord(word)

    def addWord(self, word):
        if word not in self.voc:
            self.num_words += 1
            self.voc[word] = self.num_words
            self.voc[word] = 1
            self.index2word[self.num_words] = word
        else:
            pass

    def addword(self, word):
        if word in self.voc:
            return 0
        # if len(self.voc) >= self.max_feature:
        #     raise ValueError('exceed max feature number!')
        current_max = max(self.voc.values())
        self.voc[word] = current_max + 1
        self.index2word[current_max+1] = word
        self.num_words+=1
        return 0

    def preprocessing(self,texts):
        if self.lowercase:
            return str(texts).lower()

        else:
            return str(texts)

        # 繁体转简体
        def trad2simp(text):
            return Converter('zh-hans').convert(text)

        # 连续替换，正则表达
        def multiple_replace(text, adict):
            pattern = re.compile('|'.join(adict.values()), flags=re.I)
            def matche_re(matched):
                for i in adict.keys():
                    if matched.group(i):
                        return str('<' + i + '>')
            return pattern.sub(matche_re, text)

        # 缩写替换
        def short2full(text, adict):
            pattern = re.compile('|'.join(map(re.escape, adict)), flags=re.I)
            def matche_d(matched):
                return adict[matched.group(0).upper()]  # 字典取值
            return pattern.sub(matche_d, text)

        # 同义词替换
        '''
        # 读字典, 要求: 原始词为代替换的表达,用逗号隔开,同义词为替换词
        def synonym_re(text, adict):
            for word in adict.keys():
                pattern = re.compile(word, flags=re.I)
                text = pattern.sub(adict[word], text)
                print(text)
            return text'''

        # 统一单位
        def unified(strs, dicts):
            for i in dicts.keys():
                pattern = re.compile(r'((\d+)' + i + ')', flags=re.I)
                strs = pattern.sub(r'\2' + dicts[i], strs)
            return strs

        # 文本处理
        simple_texts = trad2simp(texts)
        replace_text = multiple_replace(simple_texts, self.ReplaceDict)
        chinese_full = short2full(replace_text, self.Short2Full_dict)
        uniword = unified(chinese_full, self.Unit_dict)

        return uniword.lower()






