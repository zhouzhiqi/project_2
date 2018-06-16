import pandas as pd
import numpy as np
import time
import gc

class Count(object):
"""对特征中的不同取值按一定数据量逐块进行计数, 并保存合并好的统计结果
    
    最终保存文件的格式: 
    columns = ('click=0','click=1','total','ratio')
    index = (column + split + value, 如:'C14=10289')"""
    
    def __init__(self, ):
    """初始化参数"""
        self.data_path = '../data/project_2/'  #数据路径
        #self.file_name = self.data_path + 'train.csv'
        self.file_name = self.data_path + 'feature_add.csv'
        self.data = pd.read_csv(self.file_name, iterator=True)  #迭代读取csv
        self.chunk_size = 5e6  #每次读入数据量
        self.split = '='  # 最终保存文件index的分割符
        self.finale_click_0 = pd.Series()  #对click=0进行计数
        self.finale_click_1 = pd.Series()  #对click=1进行计数
        self.total_size = 0  ##对总的数据量进行计数
        print('counting')
        i = 0
        start_time = time.time()
        while True:
            i+=1
            #if i>10:break
            try: data_tmp = self.Loaddata()  #每次读取chunk_size条数据 
            except StopIteration: break  #读取结束后跳出循环
            click_0_tmp, click_1_tmp = self.GetClickCount(data_tmp)  #获取统计结果
            self.finale_click_0 = self.CombineSeries(self.finale_click_0, click_0_tmp)
            # 与现有click=0统计结果合并
            self.finale_click_1 = self.CombineSeries(self.finale_click_1, click_1_tmp)
            # 与现有click=1统计结果合并
            self.total_size += data_tmp.shape[0]  #累加当前数据量
            used_time = int(time.time()-start_time)  #记录统计耗时
            print('{0} data have counted, total cost time: {1}'.format(self.total_size, used_time))
            del data_tmp, click_0_tmp, click_1_tmp #及时删除, 以免内存溢出
            gc.collect()
        print('to DataFrame')
        self.ToDataFrame(self.finale_click_0, self.finale_click_1)  
        # 对已经统计好的click=0与click=1统计结果拼接整合, 并转成相应格式的DataFrame
        self.Save()  #保存计数好的文件

    def Loaddata(self, ):
    """每次读取self.chunk_size条数据"""
        return self.data.get_chunk(self.chunk_size)
    
    def ColumnIndex(self, column, index):
    """创建计数文件的索引格式"""
        return column + self.split + str(index)
    
    def GetClickCount(self, tmp):
    """对一定数据量数据进行计数"""
        click_0_index = tmp['click'] == 0
        click_0 = pd.Series()  #初始化click=0
        click_1 = pd.Series()  #初始化click=1
        for column in tmp.columns:  #按列进行计数, 并 逐列 拼接
            if column in ['id','click']: continue
            counts_tmp = self.GetCountSeries(tmp, column, click_0_index)
            # 对当前column, click=0 进行统计, 
            click_0 = pd.concat((click_0, counts_tmp))
            # 并与其column列统计结果 拼接
            counts_tmp = self.GetCountSeries(tmp, column, -click_0_index)
            # 对当前column, click=1 进行统计, 
            click_1 = pd.concat((click_1, counts_tmp))
            # 并与其column列统计结果 拼接
        return click_0, click_1
        
    def GetCountSeries(self, tmp, column, index):
    """对切片好的Series进行频率统计, 并重命名index"""
        counts = tmp[column][index].value_counts()  #
        counts.index = counts.index.map(lambda x:self.ColumnIndex(column,x))
        return counts
        
    def CombineSeries(self, a, b):
    """对传入两Series中相同索引的值累加, 不同索引的值相拼接"""
        same = a.index & b.index
        a[same] += b[same]
        unsame = b.index ^ same
        return pd.concat((a, b[unsame]))
    
    def ToDataFrame(self, click_0, click_1):
    """将点击数据转为DataFrame"""
        click_0 = click_0.to_frame(name='click=0')  #Series to DataFrame
        click_1 = click_1.to_frame(name='click=1')  #Series to DataFrame
        self.finale_counts = pd.concat((click_0,click_1),axis=1,sort=False)
        # 'click=0'与'click=0'进行拼接, 不同索引默认值为 np.nan
        self.finale_counts.fillna(0,inplace=True)  #np.nan to 0
        self.finale_counts['total'] = self.finale_counts['click=0'] + self.finale_counts['click=1']
        # 计算总的频率, 及占总数据量的比例
        self.finale_counts['ratio'] = self.finale_counts['total'] / self.total_size
        #return self.finale_counts
        
    def Save(self, file_name='Countdd.csv'):
    """将数据保存在原有数据文件夹下"""
        print('data saved in {0}'.format(self.data_path+file_name))
        self.finale_counts.to_csv(self.data_path+file_name)


def RandomSampling(file_name, output_name, mini_size=0.1, target='click', chunk_size=1e6):
    """随机对数据进行采样"""    
    
    tmp = pd.read_csv(file_name, iterator=True)  #迭代读取书据
    tmp.get_chunk(1).to_csv(output_name, mode='w', index=False)  
    #提前写入一条数据,带有表头, 以免下次写入时表头重复
    size = np.array([[0,0]])  #统计数据大小
    i=0
    while True:
        #i+=1
        #if i>10:break
        try: data = tmp.get_chunk(chunk_size)  #每次读取chunk_size条数据 
        except StopIteration: break  #读取结束后跳出循环
        mini_data = data.sample(frac=mini_size, random_state=33, ) 
        # 随机抽取mini_size(占总数据的比例)条数据
        size = np.concatenate((size,[[data.shape[0], mini_data.shape[0]]]), axis=0)
        mini_data.to_csv(output_name, mode='a', header=False, index=False)
        # 保存抽取后的数据, 不要表头(否则每次保存均有表头), 不要索引(否则单独开启一列保存索引)
        del data, mini_data #及时删除, 以免内存溢出
        gc.collect()
    size = size.sum(axis=0)
    print('file size is {0}, mini size is {1}'.format(size[0],size[1]))"


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