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
# 距离计算公式
get_distance = utils.get_distance

# 导入users和events的index索引, 以及相关信息
users_index = load(tmp_dpath+'users_index.joblib.gz')
events_index = load(tmp_dpath+'events_index.joblib.gz')
all_user = set(users_index.keys())
all_event = set(events_index.keys())
num_users = len(users_index)
num_events = len(events_index)

print('getting users information')
# 取出在train/test中出现的所有user信息
with open(dpath+'users.csv') as user_info:
    # 读入列名信息
    columns = user_info.readline().split(',')
    user_info_df=[]
    for line in user_info:
        # 若读入的user_id在train/test中出现, 
        # 添加入user_info_df
        cols = line.split(',') 
        if int(cols[0]) in all_user:  
            user_info_df.append(cols)

print('to DataFrame')
# 把生成好的user_info 转成DataFrame
user_info = pd.DataFrame(user_info_df,columns=columns,dtype=np.str)
# 把空缺值替换为np.nan
user_info.replace('',np.nan,inplace=True)
# 把user_id转换成index
user_info.index = user_info.pop('user_id').apply(lambda x:users_index[int(x)])
# 把'joinedAt'转换成日期类型
user_info_joinedAt = user_info.pop('joinedAt').astype(np.datetime64)
user_info['date'] = user_info_joinedAt.dt.date
# 把空的'birthyear'转换成np.nan
user_info['birthyear'].replace('None', np.nan, inplace=True)
user_info['birthyear'] = user_info['birthyear'].astype(np.float64)

print('Encoding . . . ')
# 对类别特征进行LabelEncoder
for c in ['locale', 'gender', 'location', 'timezone\n']:
    user_info[c] = user_info[c].astype('category').values.codes
# 对'date', 'birthyear'进行归一化处理
for c in ['date', 'birthyear']:
    min_ = user_info[c].min()
    max_ = user_info[c].max()
    user_info[c] = (user_info[c] - min_) / (max_ - min_)
# 把'birthyear'的空缺值替换为中值
bir_median = user_info['birthyear'].median()
user_info['birthyear'].replace(np.nan, bir_median, inplace=True)

print('saving . . . ')
# 保存处理好的user_info
dump(user_info, tmp_dpath+'user_info.joblib.gz', compress=('gzip',3))

"""
print('getting the distance')
# 生成数值型和类别型数据
num = user_info.loc[:, ['date', 'birthyear']]
cat = user_info.loc[:, ['locale', 'gender', 'location', 'timezone\n']]
# 生成空的distance, 默认为1
user_distance = np.ones((num_users, num_users), dtype=np.float64)
# 生成user_index
u_index = user_info.index
# 生成1/10的界限, 用于输出进度
num_users010 = num_users//10
for i,u1 in enumerate(u_index):
    # 显示进度
    if i < num_users010: print(i)
    # 对角线距离为0
    user_distance[u1,u1] = 0
    for u2 in u_index[i+1:]:
        # 计算距离, 并对称赋值
        dis = get_distance(u1,u2,num,cat)
        user_distance[u1,u2] = dis
        user_distance[u2,u1] = dis
        #print(dis)

print('saving . . . ')
# 保存处理好的user_distance
dump(user_distance, tmp_dpath+'user_distance.joblib.gz', compress=('gzip',3))
"""
pass
