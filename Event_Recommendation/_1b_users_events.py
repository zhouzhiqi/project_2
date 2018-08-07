import numpy as np
import pandas as pd

import scipy.sparse as ss
from scipy.spatial.distance import jaccard, cosine

from sklearn.externals.joblib import dump, load

import utils

# 数据路径
dpath = utils.dpath
# 数据类型
data_types = utils.data_types
# 缓存数据路径
tmp_dpath = utils.tmp_dpath

# 导入users索引
users_index = load(tmp_dpath+'users_index.joblib.gz')
# 导入events索引
events_index = load(tmp_dpath+'events_index.joblib.gz')
num_users = len(users_index)
num_events = len(events_index)

print('get users attended in events')
# 取出在总数据中出现的event
with open(dpath+'event_attendees.csv') as events_atd:
    # 生成列名(最后一个列名有'***\n')
    columns = events_atd.readline().split(',')
    events_atd_df=[]
    # 逐行读入数据
    for line in events_atd:
        # 将行以','分隔成list
        cols = line.split(',') 
        # 若读入的event_id在train/test中出现, 
        # 添加入event_test_df
        if int(cols[0]) in events_index.keys():  
            events_atd_df.append(cols)
print('to DataFrame')
# 生成参加events中的users的DF
events_atd = pd.DataFrame(events_atd_df,columns=columns,dtype=np.str)
# 把空缺值替换为np.nan
events_atd.replace('',np.nan,inplace=True)
#print(events_atd.shape)

print('to scipy.sparse')
# 只统计'yes'中的users
eventsusers = events_atd['yes'].copy()
# 把index换成event_id
eventsusers.index = events_atd.event.astype(np.int64)
# 生成空的出席稀疏矩阵{0,1}, 
# 0:不是'yes',  1:是'yes'
user_event = ss.lil_matrix((num_users, num_events), dtype=np.int8)
# 循环读入event_id
for e in eventsusers.index:
    # 如果'yes'的数据不是np.nan, 则进行统计
    if eventsusers[e] is not np.nan:
        # 生成出席该event的users列表
        user_list = np.array(eventsusers[e].split(' ')).astype(np.int64)
        # 把在train/test中出现的user, 添加入稀疏矩阵
        for u in user_list:
            try: u_index = users_index[u]
            except KeyError: continue
            else: user_event[u_index, events_index[e]] = 1

print('saving user<->event scipy.sparse')
# 保存user<->event矩阵
dump(user_event, tmp_dpath+'user_event.joblib.gz', compress=('gzip',3))
