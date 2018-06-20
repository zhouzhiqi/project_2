"""import os

print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
ust created a child process (%s).' % (os.getpid(), pid))else:
    print('I (%s) j
"""
"""
class f(object):

    def __init__(sel, tmp):
        a = 0
        for i in range(tmp*500):
            a += 1
        #print(a)

#from multiprocessing import 
from multiprocessing import Pool
#pool = multiprocessing.pool
#import multiprocessing.Pool as pool
import time

#print(dir(multiprocessing.multiprocessing))

start = time.time()
for i in range(1000,1100):
    f(i)
print('signal',time.time() - start)

start = time.time()
with Pool(10) as p:
    p.map(f, ([i for i in range(1000,1100)]))
print('multi', time.time() - start)



import tarfile
import os

def ExtractData( data_path):
    for file_name in os.listdir(data_path):
        with tarfile.open(data_path + file_name, 'r:gz') as tar: 
            tar.extractall(data_path)

ExtractData('/media/zhou/0004DD1700005FE8/AI/00/data/test/')
"""
import numpy as np
import pandas as pd
import time
import gc
import os
import scipy.sparse as ss
#from  sklearn.cross_validation  import  train_test_split 
#import xgboost as xgb
import tarfile
from multiprocessing import Pool


class test(object):

    def XgboostEncoder(self, X_train, y_train, model_path,  num_trees, deep ):
        "对xgboost输出的结点位置文件, 进行onehot"
        #model_name = 'tree{0}_deep{1}.xgboost'.format(num_trees, deep)
        #生成空白onehot矩阵, 用于赋值为1,  展开后的维数:每颗树实际有2**(deep+1)个结点, deep为模型的参数max_depth
        length = 2**(deep+1)
        leaf_index = np.zeros((X_train.shape[0], num_trees*length), dtype=np.int8)
        #转为xgb专用数据格式
        print('to xgb.DMatrix')
        xgtrain = xgb.DMatrix(X_train, label = y_train,)
        #导入模型
        xgb_model = xgb.Booster(model_file=model_path + 'tree{0}_deep{1}.xgboost'.format(num_trees, deep))
        #开始预测
        print('xgboost predict . . .')
        new_feature = xgb_model.predict(xgtrain, pred_leaf=True)  #pred_leaf=True, 输出叶子结点索引
        #对新特征onehot编码
        for i in np.arange(X_train.shape[0]):
            for tree in np.arange(num_trees):  
                #tree*length是每颗树索引的区域块, 
                #new_feature[i,tree]是该颗树的叶子结点索引
                j = tree*length + new_feature[i,tree]
                leaf_index[i, j] = 1
        return ss.csr_matrix(leaf_index)