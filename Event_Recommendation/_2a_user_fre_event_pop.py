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
user_event = load(tmp_dpath+'user_event.joblib.gz')

print('get users attended in events')
# 取出在总数据中出现的event
with open(dpath+'event_attendees.csv') as events_atd:
    # 生成列名(最后一个列名有'***\n')
    columns = events_atd.readline().split(',')
    length = len(columns)
    events_atd_df=[]
    # 逐行读入数据
    for line in events_atd:
        # 将行以','分隔成list
        cols = line.split(',') 
        if int(cols[0]) in all_event: 
            for i in range(1,length):
                if i == 4: cols[i] = cols[i][:-1]
                #print(i)
                if cols[i] == '': continue
                cols[i] = cols[i].split(' ')
                cols[i] = set(map(lambda x:int(x), cols[i]))
                #print(cols[i])
                #cols[i] = set(map(lambda x:users_index.get(int(x), np.nan), cols[i]))
                #print(cols[i]) 
            events_atd_df.append(cols)
print('to DataFrame, get [yes_num] [all_num]')
# 生成参加events中的users的DF, 空缺值为''
events_atd = pd.DataFrame(events_atd_df,columns=columns,)
# 把event_id转换成index
events_atd.index = events_atd.pop('event').apply(lambda x:events_index[int(x)])
# 统计各列的人数, 无人数设为1
events_atd_num = events_atd.applymap(lambda x:len(x))
# 参加活动的人数, 做为活动的流行程度
events_yes_num = to_0_1(events_atd_num['yes'].copy())
events_yes_num.name = 'yes_num'
events_all_num = to_0_1(events_atd_num.sum(axis=1))
events_all_num.name = 'all_num'
#events_atd = events_atd.applymap(lambda x:set(map(lambda y:users_index.get(y, -1), x)))
event_pop = pd.concat((events_yes_num, events_all_num), axis=1)

#print('saving ...')
# 保存处理好的events_yes_num
#dump(events_yes_num, tmp_dpath+'events_yes_num.joblib.gz', compress=('gzip',3))
# 保存处理好的events_all_num
#dump(events_all_num, tmp_dpath+'events_all_num.joblib.gz', compress=('gzip',3))

print('get users friends')
with open(dpath+'user_friends.csv') as users_fred:
    # 读入列名信息
    columns = users_fred.readline()[:-1].split(',')
    users_fred_df=[]
    for line in users_fred:
        # 若读入的user_id在train/test中出现, 
        # 添加入users_fred_df
        cols = line.split(',') 
        if int(cols[0]) in all_user:  
            cols[1] = cols[1][:-1].split(' ')
            cols[1] = set(map(lambda x:int(x), cols[1]))
            users_fred_df.append(cols)
print('to DataFrame, get [fre_num]')
# 把生成好的users_fred 转成DataFrame
users_freds = pd.DataFrame(users_fred_df,columns=columns,)
# 把空缺值替换为np.nan
users_freds.replace('',np.nan,inplace=True)
# 把user_id转换成index
users_freds.index = users_freds.pop('user').apply(lambda x:users_index[int(x)])
# 统计各列的人数, 转为set后, 再转为对应user_index
users_freds_num = to_0_1(users_freds['friends'].apply(lambda x:len(x)))
users_freds_num.name = 'fre_num'
#users_freds = users_freds['friends'].apply(lambda x:set(map(lambda y:users_index.get(y, -1), x)))


#print('saving . . . ')
# 保存处理好的event_distance
#dump(users_freds_num, tmp_dpath+'users_freds_num.joblib.gz', compress=('gzip',3))

print('Load train an test')
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
print('get user and event')
# 取出'user','event'
data_u_e = data_df[['user','event']].copy()
# 转换为 users_index, events_index
data_u_e['user'] = data_u_e['user'].apply(lambda x:users_index[x])
data_u_e['event'] = data_u_e['event'].apply(lambda x:events_index[x])
# 重新定义 index
data_u_e.index = np.arange(data_u_e.shape[0])

del data_df
gc.collect()

# 生成所用参数列表
confs = [
    {'name':'events_yes_num', 'insert_data':events_yes_num,},
    {'name':'events_all_num', 'insert_data':events_all_num,},
    {'name':'users_freds_num', 'insert_data':users_freds_num,},
    ]
cfs = [
    {'data':data_u_e,  'conf': conf,}
    for conf in confs
    ]
     
# 定义函数, 用于保存 users_freds_num, 
# events_all_num, events_yes_num, 
def user_event_cf(param):
    data = param['data']
    name = param['conf']['name']
    insert_data = param['conf']['insert_data']
    # 要处理的列名
    if name[0] == 'e': c = 'event'
    elif name[0] == 'u': c = 'user'

    data[name] = 0.0
    for i in data.index:
        if i%1000 == 0: print(name+'--\t', i)
        u_or_e = data.loc[i,c]
        # 添加 num/pop 到 data
        data.loc[i,name] = insert_data[u_or_e]
        #print(name+'--',cf(u, e, user_event_scores, info,))
    print(name+'--\t','saving . . . ')
    # 保存处理好的event_distance
    dump(data, tmp_dpath+'{0}.joblib.gz'.format(name), compress=('gzip',3))

with Pool(4) as p:  
    p.map(user_event_cf, cfs)

