# coding:u8
import pandas as pd
import numpy as np
import copy,gc
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score,classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import seaborn as sns
data = pd.read_pickle('features/1_stat_feature.pkl')

predictor = [ 'num_attrs',
       'num_abstract_words', 'num_alias', 'num_candidates',
       'mention_equal_subject', 'shuminghao', 'entity_common', 'len_mention',
       'mention_start', 'cos_distance', 'mse_distance',
       'type_label_mean', 'type_label_count', 'label_mean', 'label_count',
       'm_label_mean', 'm_label_count', 'rank_num_attrs',
       'rank_num_abstract_words', 'rank_num_alias', 'rank_label_mean',
       'rank_label_count', 'rank_m_label_mean', 'rank_m_label_count',
       'rank_num_candidates', 'rank_mention_equal_subject', 'rank_shuminghao',
       'rank_entity_common', 'rank_type_label_mean', 'rank_cos_distance',
       'rank_mse_distance', 'num_attrs_var', 'num_attrs_mean', 'num_attrs_max',
       'num_abstract_words_var', 'num_abstract_words_mean',
       'num_abstract_words_max', 'num_alias_var', 'num_alias_mean',
       'num_alias_max', 'label_mean_var', 'label_mean_mean', 'label_mean_max',
       'label_count_var', 'label_count_mean', 'label_count_max',
       'm_label_mean_var', 'm_label_mean_mean', 'm_label_mean_max',
       'm_label_count_var', 'm_label_count_mean', 'm_label_count_max',
       'num_candidates_var', 'num_candidates_mean', 'num_candidates_max',
       'mention_equal_subject_var', 'mention_equal_subject_mean',
       'mention_equal_subject_max', 'shuminghao_var', 'shuminghao_mean',
       'shuminghao_max', 'entity_common_var', 'entity_common_mean',
       'entity_common_max', 'cos_distance_var', 'cos_distance_mean',
       'cos_distance_max', 'mse_distance_var', 'mse_distance_mean',
       'mse_distance_max', 'type_label_mean_var', 'type_label_mean_mean',
       'type_label_mean_max']
exclude = []
# exclude = []
temp = []
for x in predictor:
    if x not in exclude:temp.append(x)
predictor2 = temp

param = {
    #'num_leaves':100,
    'objective':'binary',
    'metric':'binary_logloss',
    'metric_freq':20,
    'learning_rate': 0.05,
    'num_threads':6,
    #'min_sum_hessian_in_leaf':10,
    'boosting_type':'gbdt',
    'subsample':0.9,
    #'xgboost_dart_mode':True,
    'colsample_bytree':0.8,
    'n_estimators':1500,
    'min_child_weight':2,
    'subsample_freq':2,
    'num_leaves':64,
    'n_jobs':-1,
}
round = 0
while True:
    train = lgb.Dataset(data[predictor2],label=data['label'])

    bst = lgb.train(param, train)

    pred = bst.predict(data[predictor2])
    data['pred'] = pred
    span = list(np.linspace(0,1, 40))
    scores = []
    for i in span:
        if i<0.3:continue
        n = i
        train_data_filter = data[data.pred>n]
        submit = train_data_filter.groupby('m_id',as_index=False)['pred'].agg({'label_id_max':'idxmax'})
        m = submit.shape[0]
        n = sum(data.groupby('m_id')['label'].sum())
        hit = data.loc[submit['label_id_max'],:]['label'].sum()
        print(m,n,hit)
        acc = hit/m
        recall = hit/n
        print(i,acc,recall,2*acc*recall/(acc+recall))
        if i>0.45: break
    # 0.358974358974359 0.9101532161265269 0.9038036809815951 0.9069673356669188
    # 0.3846153846153846 0.9066765421648503 0.9012094653812445 0.9039347375083512
    # 0.41025641025641024 0.9124173245146149 0.8995267309377739 0.9059261743781665
    label_true = data[(data.label==0)&(data.pred>0.9)]
    label_false = data[(data.label==1)&(data.pred<0.1)]
    print('true number %d,false number %d' % (len(label_true), len(label_false)))
    data.loc[(data.label == 0) & (data.pred > 0.9), 'label'] = 1
    data.loc[(data.label == 1) & (data.pred < 0.1), 'label'] = 0
    data.to_pickle('data/data_round_%d' % round)
    round+=1


