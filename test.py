#-*-coding:utf-8-*-
import json
train_data = []
with open('data/raw_data/develop.json', 'r') as f:
    for line in f:
        raw = json.loads(line)
        train_data.append(raw)
originla_train_data = train_data

submit_data = []
with open('submit/result.json','r') as f:
    for line in f:
        submit_data.append(json.loads(line))

m = 0
n = 0
hit = 0
for index, s in enumerate(submit_data):
    assert s['text_id'] == originla_train_data[index]['text_id']
    assert s['text'] == originla_train_data[index]['text']
#     m+=len(s['mention_data'])
#     for item in s['mention_data']:
#         for j in originla_train_data[index]['mention_data']:
#             if item==j:
#                 hit+=1
#     for j in originla_train_data[index]['mention_data']:
#         if j['kb_id']!='NIL':
#             n+=1
# print(m,n,hit)
# acc = hit/m
# recall = hit/n
# print(hit/m,hit/n)
# print('f1',2*acc*recall/(acc+recall))