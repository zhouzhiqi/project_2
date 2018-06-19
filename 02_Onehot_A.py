# coding: utf-8

# filelist: 
#         train:40428967,   count:9449445,
#     minitrain:4042898,   more_5:1544466
# miniminitrain:404291,   more_10:645381
#    test_click:4577464
#  


import os
print(os.getcwd())
#os.chdir('/media/zhou/0004DD1700005FE8/AI/00/project_2/')
#os.chdir('E:/AI/00/project_2')
print(os.getcwd())


try: 
    from tinyenv.flags import flags
except ImportError:
    # 若在本地运行，则自动生成相同的class
    class flags(object):
        def __init__(self):
            self.file_name = 'minitrain'
            self.output_dir = '../data/project_2/models/'
            self.data_dir = '../data/project_2/Onehot_B/'#.format(self.file_name)
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
#import xgboost as xgb
import tarfile
from multiprocessing import Pool

#解压文件
def ExtractData( data_path):
    """把data_path中的tar.gz文件全部解压出来"""
    file_list = os.listdir(data_path)  #文件列表
    if 'count.csv' in file_list: return None  #若已经解压, pass
    for file_name in file_list: #把所有.tar.gz解压到data_path
        with tarfile.open(data_path + file_name, 'r:gz') as tar: 
            tar.extractall(data_path)

#保存文件中的'id'
class SaveID(object):
    """把文件中的'id'列读取出来, 并保存"""

    def __init__(self, data_path, output_path, file_name, chunksize, index=0):
        self.data_over = False
        chunksize *= 1000 #每次读取数据
        output_name = output_path + '{0}_id.csv'.format(file_name)  #输出文件的名子
        data = self.LoadData(data_path, file_name, index)  #迭代导入数据
        data.get_chunk(1).to_csv(output_name, mode='w', index=False)  #先写入一条数据, 带有header
        self.total_size = 1
        while True:
            data_tmp = self.NextChunk(data, chunksize)  #迭代读取数据
            if self.data_over: break  #如果数据读取完毕, 跳出循环
            data_tmp.to_csv(output_name, mode='a', header=False, index=False) #累加保存csv文件
            gc.collect()
        print('the size of [id] in {0}.csv is: {1} '.format(file_name, self.total_size))

    def LoadData(self, data_path, file_name, index):
        """迭代加载数据"""
        data = pd.read_csv(data_path+'{0}.csv'.format(file_name), 
                            usecols=[index], dtype=np.uint64, iterator=True)  #[index]是'id'所在的列索引
        return data

    def NextChunk(self, data, chunksize):
        """获取 chunksize 条数据"""
        #0.4M条数据用时44min, 改变chunksize基本不会影响处理速度
        try: data_tmp = data.get_chunk(chunksize)  #每次读取chunk_size条数据 
        except StopIteration: self.data_over = True  #读取结束后跳出循环
        else: 
            self.total_size += data_tmp.shape[0]  #累加当前数据量
            return data_tmp  #返回取出的数据


#合并.npz
def MergeNpz(output_path, file_name, data_begin, threshold=10,):
    """把生成的多个.npz合并成一个文件并保存"""
    #导入从0开始的文件, 为下面的合并做准备
    X_train = ss.load_npz(output_path+'{0}_X_more{1}_begin{2}.npz'.format(file_name, threshold, data_begin[0]),)
    y_train = ss.load_npz(output_path+'{0}_y_more{1}_begin{2}.npz'.format(file_name, threshold, data_begin[0]),)
    
    for begin in range(1,len(data_begin)): #循环读入文件
        #文件暂存
        X_train_tmp = ss.load_npz(output_path+'{0}_X_more{1}_begin{2}.npz'.format(
                                    file_name, threshold, data_begin[begin]),)
        y_train_tmp = ss.load_npz(output_path+'{0}_y_more{1}_begin{2}.npz'.format(
                                    file_name, threshold, data_begin[begin]),)
        X_train = ss.vstack((X_train, X_train_tmp))  #与原有 行稀疏矩阵 进行 行连接
        y_train = ss.hstack((y_train, y_train_tmp))  #与原有 列稀疏矩阵 进行 列连接
    
    print('total shape: ',X_train.shape, ' | ', y_train.shape)
    #保存最终连接完成的稀疏矩阵
    ss.save_npz(output_path+'{0}_X_more{1}'.format(file_name, threshold), X_train)
    ss.save_npz(output_path+'{0}_y_more{1}'.format(file_name, threshold), y_train)
    print('saved ^_^')
    return X_train, y_train

#读入数据并onehot
class OneHotEncoder(object):
    """对传入数据进行onehot编码, 转成scipy.sparse的稀疏矩阵, 并保存为.npz"""

    def __init__(self, param ):
        """获取参数字典, 由于多进程Pool.map只传入一个参数"""
        data_path = param['data_path']  #数据路径
        file_name = param['file_name']  #待处理文件名
        chunksize = param['chunksize']  #迭代读取数据量
        data_begin = param['data_begin']  #数据起始索引
        data_end = param['data_end']  #数据终止索引
        output_path = param['output_path']  #输出文件夹
        threshold = param['threshold']  #对频率>=threshold的特征, 进行onehot编码
        
        self.total_size = 0  #已经读取的数据量
        self.data_over = False  #数据是否读取完全

        get_index = self.GetOnehotColums(data_path, 'count', threshold)  #获取onehot的columns
        data = self.LoadData(data_path, file_name)  #迭代导入数据
        data = self.JumpData(data, data_begin, chunksize)  #跳过前data_begin条数据
        X_train, y_train = self.Train(data, get_index, data_end, chunksize)  #处理数据, onehot编码, 返回稀疏矩阵
        self.SaveNpz(X_train, y_train, output_path, file_name, data_begin, threshold)  #保存 稀疏矩阵.npz


    def GetOneHot(self, data_tmp,  get_index):
        """对数据进行OneHot编码,
        
        data_tmp: 数据
        get_index: 通过组合好的特征名, 获取索引
        return: 返回稀疏矩阵scipy.sparse, 不包含'id'"""
        # 初始化onehot数组, 全部为0, 有值置1     get_index的[-1],是用来累加那些频率小于10的特征取值
        onehot = np.zeros((data_tmp.shape[0],get_index.shape[0]), dtype=np.int8)

        # 把特征名称与特征取值结合起来, 做为get_index的索引
        for c in data_tmp.columns: 
            if c in ['id', 'click', 'hour']: continue
            data_tmp.loc[:, c] = data_tmp.loc[:, c].map(lambda x: c+'='+str(x))

        # OneHot编码, 对符合条件的位置, 赋值 1
        for i in np.arange(data_tmp.shape[0]):
            index = data_tmp.iloc[i,3:]  #取出['C1':]的values(合成过)
            for c in index:  #逐个values去get_index的索引
                try: j =  get_index[c] #找到索引, 赋值 1
                except KeyError:  #找不到索引, 说明频率小于10,
                    onehot[i, -1] += 1  #在末尾累加
                    continue   
                onehot[i, j] = 1  #对应位置赋值为1

        # 拆分时间
        split_hour = self.SplitHour(data_tmp)

        # 压缩数据, onehot编码用行压缩, label用列压缩
        X_train = ss.csr_matrix(onehot)  #转成 行 稀疏矩阵
        X_train = ss.hstack((X_train, split_hour))  #和拆分好的时间 以 列 拼接
        y_train = ss.csc_matrix(data_tmp.loc[:, 'click'].values)  #转成 列 稀疏矩阵
        #print(X_train.shape)
        return X_train, y_train

    def LoadData(self, data_path, file_name):
        """迭代加载数据"""
        data = pd.read_csv(data_path+'{0}.csv'.format(file_name),  
                            iterator=True)
        return data

    def NextChunk(self, data, chunksize):
        """获取 chunksize 条数据"""
        #0.4M条数据用时44min, 改变chunksize基本不会影响处理速度
        try: data_tmp = data.get_chunk(chunksize)  #每次读取chunk_size条数据 
        except StopIteration: self.data_over = True  #读取结束后跳出循环
        else: 
            self.total_size += data_tmp.shape[0]  #累加当前数据量
            return data_tmp  #返回取出的数据

    def SplitHour(self, data_tmp):
        """对时间进行拆分, 并onehot"""
        # 拆分后的新特征 ['workingday':1-5,6-7, 'week'1~7,  'hour':0~23,
        # 'certaintimes':0-4,5-9,10-14,15-19,20-23,] 共37列
        split_hour = np.zeros((data_tmp.shape[0], 38), dtype=np.int8)
        hour = data_tmp.loc[:, 'hour'].values %14100000 %100
        week = ((data_tmp.loc[:, 'hour'].values %14100000 //100) %7 +2) %7
        for i in np.arange(data_tmp.shape[0]):
            split_hour[i, week[i]//5] = 1  #0-1
            split_hour[i, week[i]+1] = 1   #2-8
            split_hour[i, hour[i]+9] = 1   #9-32
            split_hour[i, hour[i]//5+33] = 1  #33-37
        return ss.csr_matrix(split_hour)

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


    def GetOnehotColums(self, data_path, file_name='count', threshold=10):
        """获取OneHot编码后的列名columns"""
        print('get count ')
        count = pd.read_csv(data_path+'{0}.csv'.format(file_name), index_col=0) 
        #读取计数文件, 第一列为index (index_col=0)
        more_th = count[count['total']>=threshold].index  
        #找出频率大于 threshold 的特征名与特征值的组合, 做为OneHot的列名columns, 
        more_th = more_th.append(pd.Index(['less_threshold']))  #添加数值型特征
        get_index = pd.Series(data=np.arange(more_th.shape[0]), index=more_th, dtype=np.uint64)
        #把OneHot的列名columns, 改造成: 特征名与特征值的组合 做为 下标索引, 顺序id 做为 取值
        #简言之: 名称 索引 -> ID 索引
        del count, more_th
        gc.collect()
        return get_index

    def JumpData(self, data, data_begin, chunksize):
        """跳过前 data_begin 条数据"""
        #print('jump data')
        if data_begin <= 0: return data
        while True:
            data_tmp = self.NextChunk(data, chunksize)  #每次读取chunk_size条数据 
            if self.total_size < data_begin: continue  #如果总的数据量小于 开始时的数据量, pass
            else: break  #如果总的数据量大于 开始时的数据量, 往下进行
            gc.collect()
        print('{0} data has jumped'.format(self.total_size))
        return data

    def Train(self, data, get_index, data_end, chunksize):
        """分割数据循环进行onehot编码, 并不断拼接"""
        start_time = time.time()
        #生成第一次数据的稀疏矩阵, 为后面的拼接做准备
        data_tmp = self.NextChunk(data, chunksize)  #每次读取chunk_size条数据 
        X_train, y_train = self.GetOneHot(data_tmp,  get_index)  
        #used_time = int(time.time()-start_time)
        #print('{0} data have to OneHot, total cost time: {1}'.format(self.total_size, used_time))

        #对之后数据进行onehot编码
        while True:
            if self.total_size >= data_end: break  #先判断数据量是否足够, 若足够, 训练结束
            data_tmp = self.NextChunk(data, chunksize)  #每次读取chunk_size条数据 
            if self.data_over: break #数据读取结束, 训练结束
            X_train_tmp, y_train_tmp = self.GetOneHot(data_tmp,  get_index) 
            X_train = ss.vstack((X_train, X_train_tmp))  #与原有 行稀疏矩阵 进行 行连接
            y_train = ss.hstack((y_train, y_train_tmp))  #与原有 列稀疏矩阵 进行 列连接
            if self.total_size % (chunksize*10) ==0:
                used_time = int(time.time()-start_time)  #记录统计耗时
                print('{0} data have to OneHot, total cost time: {1}'.format(self.total_size, used_time))
            #del X_train_tmp, y_train_tmp, data_tmp #及时删除, 以免内存溢出
            gc.collect()
        return X_train, y_train


    def SaveNpz(self, X_train, y_train, output_path, file_name, data_begin, threshold=10,):
        """保存最终连接完成的稀疏矩阵"""
        print(X_train.shape, ' | ', y_train.shape)  #输出文件shape
        ss.save_npz(output_path+'{0}_X_more{1}_begin{2}'.format(file_name, threshold, data_begin), X_train)
        ss.save_npz(output_path+'{0}_y_more{1}_begin{2}'.format(file_name, threshold, data_begin), y_train)
        print('saved ^_^')






# ===========================================
# filelist: 
#         train:40428967,   count:9449445,
#     minitrain:4042898,   more_5:1544466
# miniminitrain:404291,   more_10:645381
#    test_click:4577464
# ===========================================

if __name__ == "__main__":
    
    # 解压提前压缩好的tar.gz文件, 
    # 主要用于解压上传到tinymind中的数据, 
    # 本地运行不需要解压 
    #ExtractData(FLAGS.data_dir)

    #设定参数
    file_size = 4042898  #总的数据量
    block_size = 100000  #数据块大小
    param =[dict( data_path = FLAGS.data_dir,
            file_name = FLAGS.file_name,
            chunksize = FLAGS.chunksize,  #每次处理数据的多少, 必须被block_size整除
            data_begin = XX_data_begin,
            data_end = XX_data_begin+block_size,
            output_path = FLAGS.output_dir,
            threshold = FLAGS.threshold )
            for XX_data_begin in range(0,file_size,block_size)]
    # 多进程处理onehot
    with Pool(4) as p:  #4为进程数, 可改为更大
        p.map(OneHotEncoder, param)
    
    #设定参数
    data_path = FLAGS.data_dir
    output_path = FLAGS.output_dir
    file_name = FLAGS.file_name
    data_begins = [i for i in range(0,file_size,block_size)]
    threshold = FLAGS.threshold
    chunksize = FLAGS.chunksize
    # 把生成好的.npz全部合并可以选择部分合并, 
    # 把要合并的begin 放到 list(data_begins)中, 
    # 如[0,200000,900000]做val, [100000,300000, . . . ]做训练, 
    # 注意修改生成文件的名子
    #a, b = MergeNpz(output_path, file_name, data_begins, threshold,)
    
    # 把文件中'id'列单独保存为csv, 只有test预测的时候有用
    #SaveID(data_path, output_path, file_name, chunksize ,index=0)
