class ffm(object):  
    def __init__(self, feature_num, fild_num, feature_dim_num, feat_fild_dic, learning_rate, regular_para, stop_threshold):  
        #n个特征，m个域，每个特征维度k  
        self.n = feature_num  
        self.m = fild_num  
        self.k = feature_dim_num  
        self.dic = feat_fild_dic  
        #设置超参，学习率eta, 正则化系数lamda  
        self.eta = learning_rate  
        self.lamda = regular_para  
        self.threshold = stop_threshold  
        self.w = np.random.rand(self.n, self.m , self.k) / math.sqrt(self.k)  
        #self.G是每轮梯度平方和  
        self.G = np.ones(shape = (feature_num, fild_num, feature_dim_num), dtype = np.float64) 

def train(self, tr_l, val_l, train_y, val_y, max_echo):  
    #tr_l, val_l, train_y, val_y, max_echo分别是  
    #训练集、验证集、训练集标签、验证集标签、最大迭代次数  
        minloss = 0  
        for i in range(max_echo):  
        #迭代训练，max_echo是最大迭代次数  
            L_val = 0  
            Logloss = 0  
            order = range(len(train_y))  
            #打乱顺序  
            random.shuffle(order)  
            for each_data_index in order:  
                #取出一条记录  
                tr_each_data = tr_l[each_data_index]  
                #phi()就是模型公式  
                phi = self.phi(tr_each_data)  
                #y_i是实际的标签值  
                y_i = float(train_y[each_data_index])  
                #下面计算梯度  
                g_phi = -y_i/(1 + math.exp(y_i * phi))  
                #下面开始用梯度下降法更新模型参数  
                self.sgd_para(tr_each_data, g_phi)  
                #接下来在验证集上进行检验，基本过程和前面一样。  
            for each_vadata_index, each_va_y in enumerate(val_y):  
                val_each_data = val_l[each_vadata_index]  
                phi_v = self.phi(val_each_data)  
                y_vai = float(each_va_y)  
                Logloss += -(y_vai * math.log(phi_v) + (1 - y_vai) * math.log(1 - phi_v))  
            Logloss = Logloss/len(val_y)  
                #L_val += math.log(1+math.exp(-y_vai * phi_v))  
            print("第%d次迭代, 验证集上的LOGLOSS：%f" %(i ,Logloss))  
            if minloss == 0:  
                #minloss存储最小的LOGLOSS  
                minloss = Logloss  
            if Logloss <= self.threshold:  
                #也可以认为设定阈值让程序跳断，个人需要，可以去掉。  
                print('小于阈值！')  
                break  
            if minloss < Logloss:  
                #如果下一轮迭代并没有减少LOGLOSS就break出去（early stopping）  
                print('early stopping')  
                break   
    
def phi(self, tmp_dict):  
    #样本在这要归一化，防止计算溢出  
    sum_v = sum(tmp_dict.values())  
    #首先先找到每条数据中非0的特征的索引,放到一个列表中  
    phi_tmp = 0  
    key_list = tmp_dict.keys()  
    for i in range(len(key_list)):  
        #feat_index是特征的索引,fild_index1是域的索引,value1是特征对应的值  
        feat_index1 = key_list[i]  
        fild_index1 = self.dic[feat_index1]  
        #这里除以sum_v的目的就是对这条进行归一化（将所有特征取值归到0到1之间）  
        #当然前面已经对每个特征进行归一化了（0-1）  
        value1 = tmp_dict[feat_index1] / sum_v  
        #两个非0特征两两內积  
        for j in range(i+1, len(key_list)):  
            feat_index2 = key_list[j]  
            fild_index2 = self.dic[feat_index2]  
            value2 = tmp_dict[feat_index2] / sum_v  
            w1 = self.w[feat_index1, fild_index2]  
            w2 = self.w[feat_index2, fild_index1]  
            #最终的值由多有特征组合求和得到  
            phi_tmp += np.dot(w1, w2) * value1 * value2  
    return phi_tmp  

def sgd_para(self, tmp_dict, g_phi):  
        sum_v = sum(tmp_dict.values())  
        key_list = tmp_dict.keys()  
        for i in range(len(key_list)):  
            feat_index1 = key_list[i]  
            fild_index1 = self.dic[feat_index1]  
            value1 = tmp_dict[feat_index1] / sum_v  
            for j in range(i+1, len(key_list)):  
                feat_index2 = key_list[j]  
                fild_index2 = self.dic[feat_index2]  
                value2 = tmp_dict[feat_index2] / sum_v  
                w1 = self.w[feat_index1, fild_index2]  
                w2 = self.w[feat_index2, fild_index1]  
                #更新g以及G  
                g_feati_fildj = g_phi * value1 * value2 * w2 + self.lamda * w1  
                g_featj_fildi = g_phi * value1 * value2 * w1 + self.lamda * w2  
                self.G[feat_index1, fild_index2] += g_feati_fildj ** 2  
                self.G[feat_index2, fild_index1] += g_featj_fildi ** 2  
                #math.sqrt()只能接受一个元素，而np.sqrt()可以对整个向量开根  
                self.w[feat_index1, fild_index2] -= self.eta / np.sqrt(self.G[feat_index1, fild_index2]) * g_feati_fildj  
                self.w[feat_index2, fild_index1] -= self.eta / np.sqrt(self.G[feat_index2, fild_index1]) * g_featj_fildi  
            
    
def preprocess(tr_data, te_data, cate_list, ignore_list):  
    #tr_data, te_data分别是训练集和测试集pandas的dataframe表示  
    #cate_list, ignore_list分别是类别特征列表、以及不需要的特征列表  
    #先去掉不要的特征属性  
    tr_data.drop(ignore_list, axis=1, inplace=True)  
    te_data.drop(ignore_list, axis=1, inplace=True)  
    #print(tr_data)  
    # 先从训练集tr_data分出一部分作验证集，用train_test_split  
    X, y = tr_data.drop(["id", "target"], axis=1).values, tr_data['target'].values  
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)  
    X_test = te_data.drop(["id"], axis=1).values  
    #到这里数据集从dataframe变成了array数组格式  
    tr_data.drop(["id", "target"], axis=1, inplace = True)  
    #去掉id和target后，未onehot编码前，可以先找到域的个数。  
    m = len(tr_data.columns)  
    #设定类别特征所在索引,也就是类别域对应的索引。  
    col_list = list(tr_data.columns)  
    cate_list_index = []  
    for cate_index, cate in enumerate(col_list):  
        if cate in cate_list:  
            cate_list_index.append(cate_index)  
    # 训练数据和测试数据连接一起做归一化  
    X_all = np.concatenate([X_train, X_val, X_test])  
    # 先将所有类别属性序列化  
    lb = LabelEncoder()  
    for i in cate_list_index:  
        lb.fit(X_all[:, i])  
        X_all[:, i] = lb.transform(X_all[:, i])  
        X_train[:, i] = lb.transform(X_train[:, i])  
        X_val[:, i] = lb.transform(X_val[:, i])  
        X_test[:, i] = lb.transform(X_test[:, i])  
    # 接下来对数值型数据进行归一化，这里用最大最小归一，先设置数值型特征的索引列表  
    del lb  
    #对非类别行进行归一化  
    for i in range(len(col_list)):  
        if i not in cate_list_index:  
            minv = X_all[:, i].min()  
            maxv = X_all[:, i].max()  
            delta = maxv - minv  
            X_all[:, i] = (X_all[:, i] - minv) / delta  
            X_train[:, i] = (X_train[:, i] - minv)/ delta  
            X_val[:, i] = (X_val[:, i] - minv)/ delta  
            X_test[:, i] = (X_test[:, i] - minv)/ delta  
    #进行onhot编码  
    enc = OneHotEncoder(categorical_features = cate_list_index)  
    enc.fit(X_all)  
    gc.collect()  
    #拟合完的enc，可以输出每个类别特征取值个数列表  
    m_list = list(enc.n_values_)  
    #onehot编码以后，数据就会很稀疏，直接toarray会报内存错误。coo_matix  
    X_all = enc.transform(X_all)  
    #onehot编码后，可以输出特征的个数了。onehot以后非类别的特征顺次靠后。  
    n = X_all.get_shape()[1]  
    dic = dict()  
    h = 0  
    cate_list_index = range(len(cate_list_index))  
    for i in range(m):  
        if i in cate_list_index:  
            for j in range(m_list[0]):  
                dic[h] = i  
                h += 1  
            m_list.pop(0)  
        else:  
            dic[h] = i  
            h += 1  
    X_train = enc.transform(X_train)  
    X_val = enc.transform(X_val)  
    X_test = enc.transform(X_test)  
    #接下来调用封装的函数将非0矩阵转成前面列表中套字典的形式  
    X_train = generate_li(X_train)  
    X_val = generate_li(X_val)  
    X_test = generate_li(X_test)  
    return X_train, X_val, X_test, y_train, y_val, n, m, dic  

def generate_li(m):  
    r = list(m.row)  
    c = list(m.col)  
    value = list(m.data)  
    l = [dict() for i in range(m.get_shape()[0])]  
    for i,v in enumerate(r):  
        x = c[i]  
        l[v][x] = value[i]  
    return l  