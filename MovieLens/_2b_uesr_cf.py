import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool

import scipy.sparse as ss
from scipy.spatial.distance import jaccard, cosine

from sklearn.externals.joblib import dump, load

import utils

data_dir = '../data/'
output_dir = '../output/'
get_distance = utils.get_distance
get_user_sim = utils.get_user_sim
get_movie_sim = utils.get_event_sim


class get_sim_dis(object):
    def __init__(self, k):
        self.k = k
        self.data_dir = '../data/'
        self.output_dir = '../output/'
        self.user_movie_rating = load(self.output_dir + 'user_movie_rating.joblib.gz')
        if k == 'u_s': self.get_u_s()
        elif k == 'u_d': self.get_u_d()
        elif k == 'm_s': self.get_m_s()
        elif k == 'm_d': self.get_m_d()
            
    def get_u_s(self,):
        users = load(self.output_dir + 'users.joblib.gz')
        users.index = users.pop('UserID')
        print('getting the distance')
        num_users = users.index.max()+1
        # 生成数值型和类别型数据
        users_num = users.loc[:,'Gender_0':]
        users_cat = users.loc[:,'Gender_0':]
        # 生成空的distance, 默认为1
        user_distance = np.zeros((num_users, num_users), dtype=np.float64)
        # 生成user_index
        u_index = users.index
        for i,u1 in enumerate(u_index):
            # 显示进度
            if i%100 == 0: print(self.k,'--',i)
            # 对角线距离为0
            user_distance[u1,u1] = 1
            for u2 in u_index[i+1:]:
                # 计算距离, 并对称赋值
                #sim_d = 1 - get_distance(u1,u2,users_num,users_cat)
                sim_r = get_user_sim(u1,u2,self.user_movie_rating)
                #print(sim_d,':', sim_r)
                user_distance[u1,u2] = sim_r
                user_distance[u2,u1] = sim_r
                
        dump(user_distance, self.output_dir+'{0}.joblib.gz'.format(self.k), compress=('gzip',3))

    def get_u_d(self,):
        users = load(self.output_dir + 'users.joblib.gz')
        users.index = users.pop('UserID')
        print('getting the distance')
        num_users = users.index.max()+1
        # 生成数值型和类别型数据
        users_num = users.loc[:,'Gender_0':]
        users_cat = users.loc[:,'Gender_0':]
        # 生成空的distance, 默认为1
        user_distance = np.zeros((num_users, num_users), dtype=np.float64)
        # 生成user_index
        u_index = users.index
        for i,u1 in enumerate(u_index):
            # 显示进度
            if i%100 == 0: print(self.k,'--',i)
            # 对角线距离为0
            user_distance[u1,u1] = 1
            for u2 in u_index[i+1:]:
                # 计算距离, 并对称赋值
                sim_d = 1 - get_distance(u1,u2,users_num,users_cat)
                #sim_r = get_user_sim(u1,u2,self.user_movie_rating)
                #print(sim_d,':', sim_r)
                user_distance[u1,u2] = sim_d
                user_distance[u2,u1] = sim_d
                
        dump(user_distance, self.output_dir+'{0}.joblib.gz'.format(self.k), compress=('gzip',3))
    
    def get_m_s(self,):
        movies = load(output_dir + 'movies.joblib.gz')
        movies.index = movies.pop('MovieID')
        print('getting the distance')
        num_movies = movies.index.max()+1
        # 生成数值型和类别型数据
        movies_num = movies.loc[:,'Genres_Animation':'Genres_Western']
        movies_cat = movies.loc[:,'Title_yeay':]
        # 生成空的distance, 默认为1
        movie_distance = np.zeros((num_movies, num_movies), dtype=np.float64)
        # 生成movie_index
        m_index = movies.index
        for i,m1 in enumerate(m_index):
            # 显示进度
            if i%100 == 0: print(self.k,'--',i)
            # 对角线距离为0
            movie_distance[m1,m1] = 1
            for m2 in m_index[i+1:]:
                # 计算距离, 并对称赋值
                #sim_d = 1 - get_distance(m1,m2,movies_num,movies_cat)
                sim_r = get_movie_sim(m1,m2,self.user_movie_rating)
                #print(sim_d,':',sim_r)
                movie_distance[m1,m2] = sim_r
                movie_distance[m2,m1] = sim_r
        
        dump(movie_distance, self.output_dir+'{0}.joblib.gz'.format(self.k), compress=('gzip',3))
    
    def get_m_d(self,):
        movies = load(output_dir + 'movies.joblib.gz')
        movies.index = movies.pop('MovieID')
        print('getting the distance')
        num_movies = movies.index.max()+1
        # 生成数值型和类别型数据
        movies_num = movies.loc[:,'Genres_Animation':'Genres_Western']
        movies_cat = movies.loc[:,'Title_yeay':]
        # 生成空的distance, 默认为1
        movie_distance = np.zeros((num_movies, num_movies), dtype=np.float64)
        # 生成movie_index
        m_index = movies.index
        for i,m1 in enumerate(m_index):
            # 显示进度
            if i%100 == 0: print(self.k,'--',i)
            # 对角线距离为0
            movie_distance[m1,m1] = 1
            for m2 in m_index[i+1:]:
                # 计算距离, 并对称赋值
                sim_d = 1 - get_distance(m1,m2,movies_num,movies_cat)
                #sim_r = get_movie_sim(m1,m2,self.user_movie_rating)
                #print(sim_d,':',sim_r)
                movie_distance[m1,m2] = sim_d
                movie_distance[m2,m1] = sim_d
        
        dump(movie_distance, self.output_dir+'{0}.joblib.gz'.format(self.k), compress=('gzip',3))

with Pool(4) as p:  
    p.map(get_sim_dis, ['u_s', 'u_d', 'm_s', 'm_d',])
#user_event_cf(cfs[-1])
