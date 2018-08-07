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
user_event_scores = load(tmp_dpath+'user_event_scores.joblib.gz')


print('Load Data')
# 导入数据
data_df = load(tmp_dpath+'data_df.joblib.gz')
# 分割数据
train = data_df.iloc[:15398,:].copy()
test = data_df.iloc[15398:,:].copy()
# 训练数据
y_train = train['interested']
X_train = train.drop(['event', 'user', 'interested', 'not_interested','user_id','event_id','event_cf_reco'],axis=1)
# 测试数据
X_test = test.drop(['event', 'user', 'interested', 'not_interested','user_id','event_id','event_cf_reco'],axis=1)

print('pred ... ')
# 导入序列化后的模型
mode = load(tmp_dpath+'model.joblib.gz')
# 预测值
train_preds = mode.predict_proba(X_train)[:,1]
test_preds = mode.predict_proba(X_test)[:,1]

# user event, user为索引
user_event = test.loc[:,['user','event']].copy()
user_event.index = test['user']
user_event['pred'] = test_preds

print('to submission')
# 生成空的 Series, 
submission = pd.Series(index = set(user_event.index),)
for u in set(user_event.index):
    data_tmp = user_event.loc[u,:] #关于此user的信息
    # 取出 event及pred
    event_tmp = pd.Series(data_tmp['pred'].values, index = data_tmp['event'].values)
    # 以pred预测概率进行排序, 概率大的在前
    events = event_tmp.sort_values(ascending=False).index.values
    # 合成event列表, event之间以' '分隔
    submission[u] = ' '.join(events.astype(np.str))
# 保存数据
submission.to_csv(tmp_dpath+'submission.csv')

