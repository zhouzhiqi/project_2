import numpy as np
import pandas as pd

import scipy.sparse as ss
from scipy.spatial.distance import jaccard, cosine

from sklearn.externals.joblib import dump, load

# 数据路径
dpath = '../data/'
# 缓存数据路径
tmp_dpath = '../tmp_data/'
# 数据类型
data_types = {'user':np.int64,'event':np.int64,'invited':np.int8,'interested':np.int8,'not_interested':np.int8,}


def normalization(data):
    """归一化 pd.Series"""
    min_ = data.min()
    max_ = data.max()
    return (data - min_) / (max_ - min_)

def label_encoder(data):
    """LabelEncoder pd.Series"""
    return data.astype('category').values.codes

def get_set(i, opt, all_u_e):
    """获取set(users/events)
    
    opt=='e': 返回参加某一event的所有的users
    opt=='u': 返回某一user参加的所有的event

    i: 索引值
    opt: {'e','u'}
    all_u_e: user <-> event 稀疏矩阵
    """
    if opt == 'u': # u->e
        data = all_u_e
    elif opt == 'e': # e->u
        data = all_u_e.transpose()
    # 返回对应集合
    return set(data[i,:].rows[0])

def get_users_in_event(e, all_u_e):
    """返回参加某一event的所有的users"""
    return get_set(e, 'e', all_u_e)

def get_events_for_user(u, all_u_e):
    """返回某一user参加的所有的event"""
    return get_set(u, 'u', all_u_e)


def get_distance(i1, i2, number, category):
    """基于数值型和类别型数据 计算加权后的距离
    
    分别计算i1,i2之间的cosine和jaccard距离

    i1,i2:  索引ID
    number: DataFrame, 数值型数据, 且经过归一化
    category: DataFrame, 类别数据, 且经过LabeEncoder

    return: 加权后的距离
    """
    # 计算数值型的cosine距离
    cos_sim = cosine(number.loc[i1,:], number.loc[i2,:])
    if cos_sim is np.nan: return 1
    # 计算类别型的jaccard距离
    jac_sim = jaccard(category.loc[i1,:], category.loc[i2,:])
    if jac_sim is np.nan: return 1
    return cos_sim*0.6 + jac_sim*0.4


def user_cf_dis(u, e, u_e_s, u_info,):
    """基于数值型和类别型数据得到的距离, 进行基于user的类协同过滤"""
    # 设定推荐度初始值
    ans = 0.0
    # u 平均打分
    if len(get_events_for_user(u,u_e_s))>0:
        # 如果该user给多个event打过分, 
        ave_u = u_e_s[u,:].sum()/len(get_events_for_user(u,u_e_s))  
    else:
        # 没给任何event打过分
        ave_u = 0 

    # 出席该event的user有哪些
    u_list = get_users_in_event(e,u_e_s) 
    # 生成数值型和类别型数据, 用于计算distance
    num = u_info.loc[:, ['date', 'birthyear']]
    cat = u_info.loc[:, ['locale', 'gender', 'location', 'timezone\n']]
    sims = 0
    u1 = u
    for u2 in u_list:
        # u1, u2的相似度
        sim = 1 - get_distance(u1, u2, num, cat)
        # u2 的平均打分
        if len(get_events_for_user(u2,u_e_s))>0:
            ave = u_e_s[u2,:].sum()/len(get_events_for_user(u2,u_e_s)) 
        else:
            ave = 0
        # 叠加 相似度 * 相对打分
        ans += sim * (u_e_s[u2,e]-ave)
        # 叠加 相似度
        sims += sim
    # (叠加 相似度 * 相对打分) / (叠加 相似度)
    if sims > 0:
        ans = ans/sims
    else:
        ans = 0.0
    # 加上 u 的平均打分
    return ans+ave_u

def event_cf_dis(u, e, u_e_s, e_info,):
    """基于数值型和类别型数据得到的距离, 进行基于event的类协同过滤"""
    # 设定推荐度初始值
    ans = 0.0
    sims = 0
    # 该u出席的e活动有哪些
    e_list = get_events_for_user(u,u_e_s) 
    # 生成数值型和类别型数据, 用于计算distance
    cat = e_info.loc[:, ['city', 'state', 'country', 'lat', 'lng']]
    num = e_info.loc[:, 'c_1':]
    e1 = e
    for e2 in e_list:
        # e1, e2的相似度
        sim = 1 - get_distance(e1, e2, num, cat)
        # 叠加 相似度 * 相对打分
        ans += sim * u_e_s[u,e2]
        # 叠加 相似度
        sims += sim
    # (叠加 相似度 * 相对打分) / (叠加 相似度)
    if sims>0:
        ans=ans/sims
    else:
        ans=0.0

    return ans

def get_user_sim(u1, u2, u_e_s, ):
    """基于评分矩阵的两个用户u1和u2之间的相似度
    
    u1, u2: users_index
    u_e_s: 评分矩阵
    return: 相似度
    """
    # 基于用户的协同过滤中的两个用户u1和u2之间的相似度
    #（根据两个用户对item打分的相似度）
    similarity=0.0
    # 有效的event(u1和u2均有打分的event)
    e_list = get_events_for_user(u1,u_e_s) & get_events_for_user(u2,u_e_s)
    #如果两user无共同event, 返回simlarity=0.0
    if len(e_list) < 1: 
        return similarity

    # u1的平均打分
    ru1_mean = u_e_s[u1,:].sum()/len(get_events_for_user(u1,u_e_s))  
    # u2的平均打分
    ru2_mean = u_e_s[u2,:].sum()/len(get_events_for_user(u2,u_e_s))  
    # 计算相对平均值的打分
    ru1 = 0
    ru2 = 0
    ru12 = 0
    for e in e_list:
        ru1 += (u_e_s[u1,e] - ru1_mean)**2
        ru2 += (u_e_s[u2,e] - ru2_mean)**2
        ru12 += ru1*ru2
    # 如果u1/u2对e_list中所有事件打分都是0
    if ru1*ru2 == 0:#
        return similarity
    else:
        similarity = ru12/(np.sqrt(ru1)*np.sqrt(ru2))
    
    return similarity

def user_cf_reco(u, e, u_e_s, use_less):
    """
    根据User-based协同过滤，得到event的推荐度
    基本的伪代码思路如下：
    for item i
      for every other user v that has a preference for i
        compute similarity s between u and v
        incorporate v's preference for i weighted by s into running average
    return top items ranked by weighted average
    """
    # 设定推荐度初始值
    ans = 0.0
    # u 平均打分
    if len(get_events_for_user(u,u_e_s))>0:
        # 如果该user给多个event打过分, 
        ave_u = u_e_s[u,:].sum()/len(get_events_for_user(u,u_e_s))  
    else:
        # 没给任何event打过分
        ave_u = 0 

    # 出席该event的user有哪些
    u_list = get_users_in_event(e,u_e_s) 
    
    sims = 0
    u1 = u
    for u2 in u_list:
        # u1, u2的相似度
        sim = get_user_sim(u1,u2,u_e_s)
        # u2 的平均打分
        if len(get_events_for_user(u2,u_e_s))>0:
            ave = u_e_s[u2,:].sum()/len(get_events_for_user(u2,u_e_s)) 
        else:
            ave = 0
        # 叠加 相似度 * 相对打分
        ans += sim * (u_e_s[u2,e]-ave)
        # 叠加 相似度
        sims += sim
    # (叠加 相似度 * 相对打分) / (叠加 相似度)
    if sims > 0:
        ans = ans/sims
    else:
        ans = 0.0
    # 加上 u 的平均打分
    return ans+ave_u


def get_event_sim(e1, e2, u_e_s):
    """基于评分矩阵的两个事件e1和e2之间的相似度
    
    e1, e2: users_index
    u_e_s: 评分矩阵
    return: 相似度
    """
    # 计算event e1和e2之间的相似性
    similarity=0.0
    # 同时出席e1和e2的user集合
    u_list = get_users_in_event(e1,u_e_s) & get_users_in_event(e2,u_e_s)
    #如果两event无共同user, 返回simlarity=0.0
    if len(u_list) < 1: 
        return similarity

    re1=0
    re2=0
    re12=0
    for u in u_list:
        # u的平均打分
        ru = u_e_s[u,:].sum()/len(get_events_for_user(u,u_e_s))
        # u对e1的相对打分
        r1 = u_e_s[u,e1] - ru
        # u对e2的相对打分
        r2 = u_e_s[u,e2] - ru
        re1 += r1**2
        re2 += r2**2
        re12 += r1*r2
    # 如果所有u对e1/e2打分都是0
    if re1*re2 ==0:
        return similarity
    else:
        similarity = re12/(np.sqrt(re1)*np.sqrt(re2))
    return similarity  

def event_cf_reco(u, e, u_e_s, use_less):    
    """
    根据基于物品的协同过滤，得到Event的推荐度
    基本的伪代码思路如下：
    for item i 
        for every item j that u has a preference for
            compute similarity s between i and j
            add u's preference for j weighted by s to a running average
    return top items, ranked by weighted average
    """
    # 设定推荐度初始值
    ans = 0.0
    sims = 0
    # 该u出席的e活动有哪些
    e_list = get_events_for_user(u,u_e_s) 
    e1 = e
    for e2 in e_list:
        # e1, e2的相似度
        sim = get_event_sim(e1,e2,u_e_s)
        # 叠加 相似度 * 相对打分
        ans += sim * u_e_s[u,e2]
        # 叠加 相似度
        sims += sim
    # (叠加 相似度 * 相对打分) / (叠加 相似度)
    if sims>0:
        ans=ans/sims
    else:
        ans=0.0

    return ans
