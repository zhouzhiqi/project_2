
import os
print(os.getcwd())
#os.chdir('/media/zhou/0004DD1700005FE8/AI/00/project_2/')
os.chdir('E:/AI/00/project_2')
print(os.getcwd())


try: 
    from tinyenv.flags import flags
except ImportError:
    # 若在本地运行，则自动生成相同的class
    class flags(object):
        def __init__(self):
            self.file_name = 'minitrain'
            self.output_dir = '../data/project_2/models/'
            self.data_dir = '../data/project_2/output_{0}/'.format(self.file_name)
            self.model_dir = '../data/project_2/models/'
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
from sklearn.externals import joblib


data_path = FLAGS.data_dir  
file_name = FLAGS.file_name
chunksize = FLAGS.chunksize
threshold = FLAGS.threshold
output_path = FLAGS.output_dir
deep = FLAGS.max_depth + 1   #实际树的深度为 max_depth+1
num_trees = FLAGS.num_trees  #树的数量
model_path = FLAGS.model_dir

def XGBPredict(X_train, y_train, model_path, num_trees, deep):

    #转为xgb专用数据格式
    print('to xgb.DMatrix')
    xgtrain = xgb.DMatrix(X_train, label = y_train,)
    #导入模型
    xgb_model = xgb.Booster(model_file=model_path + 'tree{0}_deep{1}.xgboost'.format(num_trees, deep))
    #开始预测
    print('xgboost predict . . .')
    train_preds = xgb_model.predict(xgtrain)
    return train_preds

def LRPredict(X_train, y_train, model_path):
    #保存模型
    lr_model = joblib.load(model_path+'LR.sklearn')
    #进行评价
    print('LR.sklearn predict . . .')
    train_preds = lr_model.predict_proba(X_train)[:,1]
    return train_preds



#导入数据
print('Load Data')
X_train = ss.load_npz(data_path+'{0}_X_more{1}.npz'.format(file_name, threshold), )
y_train = ss.load_npz(data_path+'{0}_y_more{1}.npz'.format(file_name, threshold), )
y_train = y_train.toarray().astype(np.float32)[0]
# 先转成np.array, 把数据类型转为np.float32(此时为2维数组shape(1,n)), 转为1-D np.arrar
# 2维数组shape(m,n)适用于多分类问题, 在二分类中不适用

#获得预测结果
train_preds_xgb = XGBPredict(X_train, y_train, model_path, num_trees, deep)
train_preds_llr = LRPredict(X_train, y_train, model_path)
#取平均值
train_preds = (train_preds_xgb + train_preds_llr) / 2

#合并test_id与predict
submission = pd.read_csv(data_path + '{0}_id.csv'.format(file_name), dtype=np.uint64)
submission.insert(loc=1, column='click', value=train_preds)

print('to submission.csv')
#保存预测好的文件
submission.to_csv(output_path + 'submission.csv', index=False)