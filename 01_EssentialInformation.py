import pandas as pd
import numpy as np
import time
import gc

class EssentialInformation(object):
    """查看数据的基本信息,分布,以及直方图
    
    传入参数i(int)为某列的索引号"""
    
    
    def __init__(self, i):
        """导入i对应的特征"""
        data_path = '../data/project_2'
        train_path = os.path.join(data_path, 'train.csv')
        self.tmp = pd.read_csv(train_path,usecols=[i])
        self.name = self.tmp.columns[0]
    
    def NAN(self,):
        """查看i对应特征的空缺值"""
        print('the NAN of {0}'.format(self.name))
        print('----------------------------------')
        print(self.tmp.isnull().sum())
        
    def describe(self,):
        """查看i对应特征的数据类型及通用的描述"""
        print('the describe of {0}'.format(self.name))
        print('----------------------------------')
        print(self.tmp.info())
        print('----------------------------------')
        print(self.tmp.describe())
        
    def counts(self,):
        """查看i对应特征值出现的频率"""
        print('the counts of {0}'.format(self.name))
        print('----------------------------------')
        print(self.tmp[self.name].value_counts())

    def hist(self,):
        """查看i对应特征的直方图"""
        counts = self.tmp[self.name].value_counts()
        counts_len = counts.shape[0]
        print('特征[{0}]共有{1}个分类'.format(self.name, counts_len))
        if counts_len>12:
            print('对出现次数最多的前12个分类进行可视化')
            if self.tmp.dtypes.values=='object':
                plt.plot(counts[:12])
                plt.xticks(rotation=90)
            else:
                index = self.tmp[self.name].isin(counts[:12].index)
                plt.hist(self.tmp[self.name][index], bins=12)
        else:
            plt.hist(self.tmp[self.name], bins=counts_len)
        plt.show()

# Random sampling
a = Count()