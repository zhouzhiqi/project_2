import pandas as pd
import numpy as np
import time
import gc


class FeatureAndTarget(object):
    """"查看数据与click的关系
    
    传入参数i(int)为某列的索引号"""
    
    
    def __init__(self, i, target=target):
        """"导入i对应的特征"""
        data_path = '../data/project_2'
        train_path = os.path.join(data_path, 'train.csv')
        self.tmp = pd.read_csv(train_path,usecols=[i])
        self.name = self.tmp.columns[0]
        self.counts = self.tmp[self.name].value_counts()
        self.counts_len = self.counts.shape[0]
        self.tmp.insert(loc=0, column='click', value=target)
        index = self.tmp[self.name].isin(self.counts[:12].index)
        self.tmp = self.tmp[index]
        print('特征[{0}]与click的图像为:'.format(self.name))
        print('-----------------------------------')
        if self.counts_len>12:
            self.counts_len = 24
            print('对出现次数最多的前12个分类进行可视化')
            if self.tmp[self.name].dtypes=='object':
                self.plot()
            else:
                self.hist()
        else:
            if self.tmp[self.name].dtypes=='object':
                self.plot()
            else:
                self.hist()
        
    def hist(self,):
        """"查看i对应特征的直方图"""
        _, ax = plt.subplots()
        ax.hist(self.tmp[self.name][self.tmp['click']==1],
                    bins = self.counts_len,
                    color = '#539caf',
                    label = 'click',
                    alpha = 1)
        ax.hist(self.tmp[self.name][self.tmp['click']==0],
                    bins = self.counts_len,
                    color = '#7663b0',
                    label = 'no click',
                    alpha = 0.5)
        ax.legend(loc = 'best')
        plt.show()

    def plot(self,):
        """"查看i对应特征的直方图"""
        _, ax = plt.subplots()
        ax.plot(self.tmp[self.name][self.tmp['click']==1].value_counts()[:12],
                    color = '#539caf',
                    label = 'click',)
        ax.plot(self.tmp[self.name][self.tmp['click']==0].value_counts()[:12],
                    color = '#7663b0',
                    label = 'no click',)
        ax.legend(loc = 'best')
        plt.xticks(rotation=90)
        plt.show()

