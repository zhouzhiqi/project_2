
# coding: utf-8


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


# In[2]:


import numpy as np
import pandas as pd
import time
import gc
import os
import scipy.sparse as ss
#from  sklearn.cross_validation  import  train_test_split 
import xgboost as xgb



# In[3]:


data_path = FLAGS.data_dir  
file_name = FLAGS.file_name
chunksize = FLAGS.chunksize
threshold = FLAGS.threshold
output_path = FLAGS.output_dir
deep = FLAGS.max_depth + 1   #实际树的深度为 max_depth+1
num_trees = FLAGS.num_trees  #树的数量


# # 导入与分割数据

# In[4]:

#data_path = '../data/project_2/test_click/'
#导入数据
print('Load Data')
X_train = ss.load_npz(data_path+'{0}_X_more{1}.npz'.format(file_name, threshold), )
y_train = ss.load_npz(data_path+'{0}_y_more{1}.npz'.format(file_name, threshold), )
y_train = y_train.toarray().astype(np.float32)[0]



from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(solver='sag', n_jobs=-1)


print('training . . . ')
lr.fit(X_train, y_train, )

train_preds = lr.predict_proba(X_train)
train_predictions = np.around(train_preds)
label = y_train



from sklearn.metrics import accuracy_score,log_loss


train_accuracy = accuracy_score(label, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
train_log_loss = log_loss(y_train, train_preds)
print ("Train log_loss: " , train_log_loss)

exit()

"""
from sklearn.cross_validation import cross_val_score
loss = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_log_loss')
print('logloss of each fold is: ',-loss)
print('cv logloss is:', -loss.mean())


# ## Logistic回归调优

# GridSearchCV, 直接暴力搜索

# In[55]:

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#需要调优的参数
# 请尝试将L1正则和L2正则分开，并配合合适的优化求解算法（slover）
#tuned_parameters = {'penalty':['l1','l2'],
#                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
#                   }
penaltys = ['l1','l2']
Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
tuned_parameters = dict(penalty = penaltys, C = Cs)

lr_penalty= LogisticRegression()
grid= GridSearchCV(lr_penalty, tuned_parameters,cv=5, scoring='neg_log_loss')
grid.fit(X_train,y_train)


# In[56]:

grid.cv_results_


# In[57]:

print(-grid.best_score_)
print(grid.best_params_)


# 最好的是L2正则, 对应的C=0.1, λ=1/C=10, 说明模型还是有一定的复杂度的.
# 加正则后的best_score_=0.454345290494, 对比默认的0.45642415636稍微好一点

# In[58]:

# plot CV误差曲线
test_means = grid.cv_results_[ 'mean_test_score' ]
test_stds = grid.cv_results_[ 'std_test_score' ]
train_means = grid.cv_results_[ 'mean_train_score' ]
train_stds = grid.cv_results_[ 'std_train_score' ]


# plot results
n_Cs = len(Cs)
number_penaltys = len(penaltys)
test_scores = np.array(test_means).reshape(n_Cs,number_penaltys)
train_scores = np.array(train_means).reshape(n_Cs,number_penaltys)
test_stds = np.array(test_stds).reshape(n_Cs,number_penaltys)
train_stds = np.array(train_stds).reshape(n_Cs,number_penaltys)

x_axis = np.log10(Cs)
for i, value in enumerate(penaltys):
    #pyplot.plot(log(Cs), test_scores[i], label= 'penalty:'   + str(value))
    plt.errorbar(x_axis, test_scores[:,i], yerr=test_stds[:,i] ,label = penaltys[i] +' Test')
    plt.errorbar(x_axis, train_scores[:,i], yerr=train_stds[:,i] ,label = penaltys[i] +' Train')
    
plt.legend()
plt.xlabel( 'log(C)' )                                                                                                      
plt.ylabel( 'neg-logloss' )
#plt.savefig('LogisticGridSearchCV_C.png' )

plt.show()# plot CV误差曲线


# ## SVC_linear

# In[68]:

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#需要调优的参数
# 请尝试将L1正则和L2正则分开，并配合合适的优化求解算法（slover）
#tuned_parameters = {'penalty':['l1','l2'],
#                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
#                   }


Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

tuned_parameters = dict(C = Cs)

lrSVC_penalty= SVC(kernel='linear',probability=True)

grid= GridSearchCV(lrSVC_penalty, param_grid=tuned_parameters, cv=5, scoring='neg_log_loss')
grid.fit(X_train,y_train)


# In[69]:

grid.cv_results_


# In[70]:

print(-grid.best_score_)
print(grid.best_params_)


# ## SVC_rbf

# In[72]:

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#需要调优的参数
# 请尝试将L1正则和L2正则分开，并配合合适的优化求解算法（slover）
#tuned_parameters = {'penalty':['l1','l2'],
#                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
#                   }


C_s = np.logspace(-1, 3, 5)# logspace(a,b,N)把10的a次方到10的b次方区间分成N份 
gamma_s = np.logspace(-2, 2, 5) 

tuned_parameters = dict(gamma=gamma_s, C = C_s)

lrSVC_penalty= SVC(kernel='rbf',probability=True)

grid= GridSearchCV(lrSVC_penalty, param_grid=tuned_parameters, cv=5, scoring='neg_log_loss')
grid.fit(X_train,y_train)


# In[73]:

grid.cv_results_


# In[74]:

print(-grid.best_score_)
print(grid.best_params_)


# # 结论

#     rbf核的log_loss=0.468160621347, 网格搜索结果为{'C': 1.0, 'gamma': 0.01}
#     linear核的log_loss=0.455566872027, 网格搜索结果为{'C': 0.1}
#     调参后的logistic回归的log_loss=0.454345290494, 网格搜索结果为{'C': 0.1, 'penalty': 'l2'}
#     默认的logistic回归的log_loss=0.45642415636

# 最小的log_loss=0.454345290494, 是调参后的logistic回归的, 对应的网格搜索结果为{'C': 0.1, 'penalty': 'l2'}

# In[ ]:

"""

pass

