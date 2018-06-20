# coding: utf-8

# filelist: 
#         train:40428967,  
#     minitrain:4042898,  
# miniminitrain:404291,  
#    test_click:4577464
#  

#import os
#os.chdir('/media/zhou/0004DD1700005FE8/AI/00/project_2/')
#os.chdir('E:/AI/00/project_2')

try: 
    from tinyenv.flags import flags
except ImportError:
    # 若在本地运行，则自动生成相同的class
    class flags(object):
        def __init__(self):
            self.file_name = 'minitrain'
            self.onehot_name = 'Onehot_A'
            self.data_dir = '../data/project_2/data/{0}/'.format(self.onehot_name)
            self.output_dir = '../data/project_2/output/{0}/'.format(self.onehot_name)
            self.model_dir = '../data/project_2/models/{0}/'.format(self.onehot_name)


#实例化class
FLAGS = flags()

#导入工具包
import numpy as np
import pandas as pd
import time
import gc
import os
import scipy.sparse as ss
#from  sklearn.cross_validation  import  train_test_split 
#import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,log_loss
from sklearn.externals import joblib
from sklearn.ensemble import VotingClassifier

#定义参数
data_path = FLAGS.data_dir  
file_name = FLAGS.file_name
chunksize = FLAGS.chunksize
threshold = FLAGS.threshold
output_path = FLAGS.output_dir
deep = FLAGS.max_depth + 1   #实际树的深度为 max_depth+1
num_trees = FLAGS.num_trees  #树的数量
model_path = FLAGS.model_dir

#导入数据
print('Load Data')
X_train = ss.load_npz(data_path+'{0}_X_more{1}.npz'.format(file_name, threshold), )
y_train = ss.load_npz(data_path+'{0}_y_more{1}.npz'.format(file_name, threshold), )
y_train = y_train.toarray().astype(np.float32)[0]


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
        
deep = 8   #实际树的深度为 max_depth+1
num_trees = 30  #树的数量
c = 0.5  #正则化惩罚系数

#导入模型
#lr = LogisticRegression(multi_class='ovr', penalty='l2', solver='sag', C=c, n_jobs=-1)
rf = DecisionTreeClassifier( criterion='entropy', min_samples_split=4,
                            max_depth=8,  random_state=33, )
#开始训练
start_time = time.time()
print('training . . . ')
#lr.fit(X_train, y_train, )
rf.fit(X_train, y_train, )
print('cost time:{0}'.format(int(time.time() - start_time)))

#保存模型
#joblib.dump(lr, model_path+'LR_sklearn.model')

#进行评价
#train_preds = lr.predict_proba(X_train)[:,1]
#train_predictions = np.around(train_preds)

#train_accuracy = accuracy_score(y_train, train_predictions)
#print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
#train_log_loss = log_loss(y_train, train_preds)
#print ("Train log_loss: " , train_log_loss)


#进行评价
train_preds = rf.predict_proba(X_train)[:,1]
train_predictions = np.around(train_preds)

train_accuracy = accuracy_score(y_train, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
train_log_loss = log_loss(y_train, train_preds)
print ("Train log_loss: " , train_log_loss)
