import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool

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
# 距离计算公式
get_distance = utils.get_distance
to_0_1 = utils.normalization
to_cat = utils.label_encoder

# 导入users和events的index索引, 以及相关信息
users_index = load(tmp_dpath+'users_index.joblib.gz')
events_index = load(tmp_dpath+'events_index.joblib.gz')
all_user = set(users_index.keys())
all_event = set(events_index.keys())
num_users = len(users_index)
num_events = len(events_index)
user_event_scores = load(tmp_dpath+'user_event_scores.joblib.gz')
user_event = load(tmp_dpath+'user_event.joblib.gz')
#user_distance = load(tmp_dpath+'user_distance.joblib.gz')
#event_distance = load(tmp_dpath+'event_distance.joblib.gz')
user_info = load(tmp_dpath+'user_info.joblib.gz')
event_info = load(tmp_dpath+'event_info.joblib.gz')


# 读入训练数据
train = pd.read_csv(dpath+'train.csv',dtype=data_types, index_col=['timestamp'])
# 以时间类型数据为index
train.index = train.index.astype(np.datetime64)
# 读入测试数据
test = pd.read_csv(dpath+'test.csv',dtype=data_types, index_col=['timestamp'])
# 以时间类型数据为index
test.index = test.index.astype(np.datetime64)
# 拼接数据
data_df = pd.concat((train, test), axis=0)

# 取出'user','event'
data_u_e = data_df[['user','event']].copy()
# 转换为 users_index, events_index
data_u_e['user'] = data_u_e['user'].apply(lambda x:users_index[x])
data_u_e['event'] = data_u_e['event'].apply(lambda x:events_index[x])
# 重新定义 index
data_u_e.index = np.arange(data_u_e.shape[0])

del data_df
gc.collect()

confs = [
    {'name':'user_cf_dis', 'cf':utils.user_cf_dis, 'info':user_info,},
    {'name':'event_cf_dis', 'cf':utils.event_cf_dis, 'info':event_info,},
    {'name':'user_cf_reco', 'cf':utils.user_cf_reco, 'info':None,},
    {'name':'event_cf_reco', 'cf':utils.event_cf_reco, 'info':None,},
    ]
cfs = [
    {'u_e_s':user_event_scores, 'data':data_u_e, 'conf': conf,}
    for conf in confs
    ]
     

def user_event_cf(param):
    u_e_s = param['u_e_s']
    data = param['data']

    name = param['conf']['name']
    cf = param['conf']['cf']
    info = param['conf']['info']

    data[name] = 0.0
    for i in data.index:
        if i%1000 == 0: print(name+'--\t', i)
        u = data.loc[i,'user']
        e = data.loc[i,'event']
        data.loc[i,name] = cf(u, e, u_e_s, info, )
        #print(name+'--',cf(u, e, user_event_scores, info,))
    print('saving . . . ')
    # 保存处理好的event_distance
    dump(data, tmp_dpath+'{0}.joblib.gz'.format(name), compress=('gzip',3))

#with Pool(4) as p:  
#    p.map(user_event_cf, cfs)
user_event_cf(cfs[-1])
