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

print('Load train test')
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

print('encoding')
#data_df['user_id'] = data_df['user'].apply(lambda x:users_index[int(x)])
#data_df['event_id'] = data_df['event'].apply(lambda x:events_index[int(x)])
# 生成新特征
data_df['day'] = to_0_1(to_cat(data_df.index.day)) 
data_df['month'] = to_0_1(to_cat(data_df.index.month)) 
data_df['weekday'] = to_0_1(to_cat(data_df.index.weekday)) 
data_df['date'] = to_0_1(data_df.index.date).astype(np.float64) 
# 重新定义 index
data_df.index = np.arange(data_df.shape[0])
# 待添加的文件名
confs = [
    {'name':'user_cf_dis', },
    {'name':'event_cf_dis', },
    {'name':'user_cf_reco', },
#    {'name':'event_cf_reco', },
    {'name':'events_yes_num', },
    {'name':'events_all_num', },
    {'name':'users_freds_num', },
    ]
# 导入数据 并添加到train/test
for conf in confs:
    name = conf['name']
    print('merging data:\t', name)
    data = load(tmp_dpath+'{0}.joblib.gz'.format(name))
    data_df[name] = data[name] 
    
print('saving ...')
# 保存最终的train/test
print(data_df[:5])
dump(data_df, tmp_dpath+'data_df.joblib.gz', compress=('gzip',3))

