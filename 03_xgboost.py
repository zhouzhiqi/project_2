# coding: utf-8

# |名称|含义|
# | : | : |
# |id | ad identifier
# |click | 1是点击 0/1 for non-click/click
# |hour | 时间, YYMMDDHH, so 14091123 means 2014年09月11号23:00 UTC.
# |C1 | 未知分类变量 anonymized categorical variable
# |banner_pos|标语,横幅
# |site_id|网站ID号
# |site_domain|网站 域?
# |site_category|网站类别
# |app_id|appID号
# |app_domain|应用 域?
# |app_category|应用类别
# |device_id|设备ID号
# |device_ip|设备ip地址
# |device_model|设备型号, 如iphone5/iphone4
# |device_type|设备类型, 如智能手机/平板电脑
# |device_conn_type|连接设备类型
# |C14-C21|未知分类变量 anonymized categorical variables


try: 
    from tinyenv.flags import flags
except ImportError:
    # 若在本地运行，则自动生成相同的class
    class flags(object):
        def __init__(self):
            self.output_dir = '../data/project_2/models/'
            self.data_dir = '../data/project_2/output_minitrain/'
            self.file_name = 'minitrain'
            self.chunksize = 1e3
            self.threshold = 10
            self.data_begin = 0
            self.data_end = 1e5
            self.id_index = 0
            self.num_trees = 30
            self.max_depth = 8

#实例化class
FLAGS = flags()

import numpy as np
import pandas as pd
import time
import gc
import os
import scipy.sparse as ss
#from  sklearn.cross_validation  import  train_test_split 
import xgboost as xgb


data_path = FLAGS.data_dir  # + '../data/project_2/'
file_name = FLAGS.file_name
chunksize = FLAGS.chunksize
threshold = FLAGS.threshold
output_path = FLAGS.output_dir

#导入数据
print('Load Data')
X_train = ss.load_npz(data_path+'{0}_X_more{1}.npz'.format(file_name, threshold), )
y_train = ss.load_npz(data_path+'{0}_y_more{1}.npz'.format(file_name, threshold), )
y_train = y_train.toarray().astype(np.float32)[0]
# 先转成np.array, 把数据类型转为np.float32(此时为2维数组shape(1,n)), 转为1-D np.arrar
# 2维数组shape(m,n)适用于多分类问题, 在二分类中不适用

#转为xgb专用数据格式
print('to xgb.DMatrix')
xgtrain = xgb.DMatrix(X_train, label = y_train,)

#设置参数, 开始训练
print('training . . .')

param = dict(
        learning_rate =0.4, 
        booster='gbtree',
        n_estimators=100,  
        max_depth=FLAGS.max_depth, #树的深度
        min_child_weight=4,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        objective= 'binary:logistic' ,
        eta=0.4,
        silent=0,
        eval_metric='logloss',
        seed=3)
deep = FLAGS.max_depth + 1   #实际树的深度为 max_depth+1
num_trees = FLAGS.num_trees  #树的数量

#调用cv函数
#bst_train = xgb.cv(param, xgtrain, num_trees, nfold=3, stratified=True)
#打印训练信息
#print(bst_train)

#训练模型
bst_train = xgb.train(param, xgtrain, num_trees, )
#保存模型
bst_train.save_model(output_path + 'xgb_tree{0}_deep{1}'.format(num_trees, deep))



"""
#生成xgb处理后的高阶特征
new_feature = bst_train.predict(xgtrain, pred_leaf=True)
print('shape is:', new_feature.shape)
print(new_feature.max(), new_feature.min())
"""

from sklearn.metrics import accuracy_score,log_loss

train_preds = bst_train.predict(xgtrain)
train_predictions = np.around(train_preds)
label = xgtrain.get_label()
train_accuracy = accuracy_score(label, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
train_log_loss = log_loss(y_train, train_preds)
print ("Train log_loss: " , train_log_loss)

print('over-----')

pass