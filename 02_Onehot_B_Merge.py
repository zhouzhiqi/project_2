import pandas as pd
import numpy as np
import time
import gc

class Feature_Combine(object):
    
    
    def __init__(self,):
        self.data_path = '../data/project_2/'
        self.file_name = self.data_path + 'train.csv'
        #self.file_name = self.data_path + 'miniminitrain.csv'
        self.data = pd.read_csv(self.file_name, iterator=True)  #迭代读取csv
        self.chunk_size = 1e6  #每次读入数据量
        self.split = '_'
        self.total_size = 0
        self.SaveHeader()
        i = 0
        start_time = time.time()
        while True:
            i+=1
            #if i>10:break
            try: self.data_tmp = self.data.get_chunk(self.chunk_size)  #每次读取chunk_size条数据 
            except StopIteration: break  #读取结束后跳出循环
            self.total_size += self.data_tmp.shape[0]  #累加当前数据量
            used_time = int(time.time()-start_time)  #记录统计耗时
            print('{0} data have counted, total cost time: {1}'.format(self.total_size, used_time))
            self.CombineAll()
            self.Save()


    def CombineAll(self,):
        end = np.array([3,5,8,11,16,24])
        columns = {3:'C1BP', 5:'site', 
                 8:'app', 11:'device', 16:'C14to21'}
        
        self.finaldata = self.Combine(self.Loaddata(3), 4)
        self.finaldata = self.finaldata.to_frame(name=columns[3])

        for i in np.arange(5,23):
            if i in end: 
                feature = self.Loaddata(i)
                name = columns[i]
            feature = self.Combine(feature, i+1)
            if i+2 in end:
                self.Insert(name, feature)
                
        for i,name in enumerate(['id','click','hour']):
            self.Insert(name, self.Loaddata(i))
        #self.Save()
        
    def Insert(self, name, values):
        self.finaldata.Insert(loc=0, column=name, value=values)
        
    def Loaddata(self, index):
        tmp = self.data_tmp.iloc[:, index]
        return tmp.map(str)#.iloc[:,0]

    def Combine(self, data, col, split='_'):
        tmp = self.Loaddata(col)
        data = data+self.split+tmp
        return data
    
    def SaveHeader(self, file_name='feature_add.csv'):
        """将数据保存在原有数据文件夹下"""
        header = pd.DataFrame(columns=['hour','click','id','C14to21','device','app','site','C1BP'])
        header.to_csv(self.data_path+file_name, 
                      mode='w', header=True, index=False)
        
    def Save(self, file_name='feature_add.csv'):
        """将数据保存在原有数据文件夹下"""
        self.finaldata.to_csv(self.data_path+file_name, 
                              mode='a', header=False, index=False)


    
