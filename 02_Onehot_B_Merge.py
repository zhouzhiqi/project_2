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



class Onehot_B_Merge(object):
    
    
    def __init__(self, data_path, file_name, output_path, chunksize, split):
        self.data_path = '../data/project_2/'
        self.file_name = '{0}.csv'.format(file_name)

        chunk_size = chunksize  #每次读入数据量
        self.data_over = False  #数据是否读入结束
        self.total_size = 0  #已经读入数据总量
        self.SaveHeader(output_path, file_name)  #首先保存header

        i = 0
        data = self.LoadData(data_path, file_name)
        start_time = time.time()
        while True:
            #i+=1
            #if i>10:break
            data_tmp = self.NextChunk(data, chunk_size)  #获取数据块
            if self.data_over: break  #如果数据遍历完毕, 结束
            merge_df = self.MergeWithSplit(data_tmp, split)
            self.Save(merge_df, output_path, file_name)
            used_time = int(time.time()-start_time)  #记录统计耗时
            print('{0} data have merged, total cost time: {1}'.format(self.total_size, used_time))



    def MergeWithSplit(self, data, split='&'):
        #print('mergeing . . .')
        self.merge_df = data.loc[:,['id','click','hour']]  #['id','click','hour']不进行处理

        value = self.Merge(data, ['C1','banner_pos'])
        self.DFInsertValue(self.merge_df, 'C1BP', value)

        value = self.Merge(data, ['site_id','site_domain','site_category'])
        self.DFInsertValue(self.merge_df, 'site', value)

        value = self.Merge(data, ['app_id','app_domain','app_category'])
        self.DFInsertValue(self.merge_df, 'app', value)

        value = self.Merge(data, ['device_id','device_ip'])
        self.DFInsertValue(self.merge_df, 'dev_idip', value)

        value = self.Merge(data, ['device_model', 'device_type', 'device_conn_type'])
        self.DFInsertValue(self.merge_df, 'dev_types', value)

        value = self.Merge(data, ['C14','C17'])
        self.DFInsertValue(self.merge_df, 'C1417', value)

        value = self.Merge(data, ['C15','C16'])
        self.DFInsertValue(self.merge_df, 'C1516', value)

        value = self.Merge(data, ['C18','C20'])
        self.DFInsertValue(self.merge_df, 'C1820', value)

        value = self.Merge(data, ['C19','C21'])
        self.DFInsertValue(self.merge_df, 'C1921', value)

        return self.merge_df

    def Merge(self, data, cols, split='&'):
        """结合方法, 字符串以分隔符相拼接"""
        tmp = self.LoadFromCol(data, cols[0])
        for col in cols[1:]:
            tmp = tmp + split + self.LoadFromCol(data, col)
        ###可以在此处添加hash转化, int32 为 -2*10^9 ~ 2*10^9
        return tmp.map(lambda x:hash(x)%1e9).astype(np.int32)

    def DFInsertValue(self, data_df, name, values):
        """在DataFrame的末尾插入values"""
        length = data_df.shape[1]  #DF长度
        return data_df.insert(loc=length, column=name, value=values)
        
    def LoadData(self, data_path, file_name):
        """迭代加载数据"""
        print('load data')
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

    def LoadFromCol(self, data, col_name):
        """导入对应列名的数据, 并转为字符串"""
        tmp = data.loc[:, col_name]  #读入数据
        return tmp.map(str)  #转为字符串

    def SaveHeader(self, output_path, file_name, ):
        """将数据保存在原有数据文件夹下"""
        header = pd.DataFrame(columns=['id','click','hour','C1BP',
                                        'site','app','dev_idip','dev_types',
                                        'C1417','C1516','C1820','C1921',])
        header.to_csv(output_path + 'Onehot_B_{0}.csv'.format(file_name), 
                      mode='w', header=True, index=False)
        
    def Save(self, data, output_path,  file_name, ):
        """将数据累加保存在原有数据文件夹下"""
        data.to_csv(output_path + 'Onehot_B_{0}.csv'.format(file_name), 
                    mode='a', header=False, index=False)



Onehot_B_Merge(file_name = FLAGS.file_name,
data_path = FLAGS.data_dir,
output_path = FLAGS.output_dir,
chunksize = FLAGS.chunksize,
split = FLAGS.split)
