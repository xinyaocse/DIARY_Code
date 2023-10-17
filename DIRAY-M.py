# -*- coding: utf-8 -*-


# 导入基本库
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import  random

def clip(train_part,tr_y):
    combined_array = list(zip(train_part, tr_y))

    # 将打乱后的元组列表拆分回两个数组
    shuffled_array1, shuffled_array2 = zip(*combined_array)

    X_train, X_test, y_train, y_test = train_test_split(
        shuffled_array1, shuffled_array2, test_size=0.4)  # 按照比例划分数据集为训练集与测试集

    return X_train, y_train, X_test, y_test

def random_reverse_array(arr):
    """
    将输入的 numpy array 随机排列，并将元素顺序进行颠倒
    """
    # 将数组进行随机排列
    random.shuffle(arr)
    return arr
def my_read_data(dataset,length):
    kcenter_list = []
    uncertain_list = []
    random_list = []
    dfal_list = []
    dfal_k_list = []
    knockoff=[]
    for i in range(1, 41):
        temp = np.load("./data_generate/svm_result/"+str(dataset)+"/feature/"+str(length)+"/kcenter/" + str(i) + ".npy", allow_pickle=True).flatten()
        kcenter_list.append(temp)
    for i in range(1, 201):
        temp = np.load("./data_generate/svm_result/"+str(dataset)+"/feature/"+str(length)+"/random/" + str(i) + ".npy", allow_pickle=True).flatten()
        random_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./data_generate/svm_result/"+str(dataset)+"/feature/"+str(length)+"/uncertainty/" + str(i) + ".npy", allow_pickle=True).flatten()
        uncertain_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./data_generate/svm_result/"+str(dataset)+"/feature/"+str(length)+"/dfal/" + str(i) + ".npy", allow_pickle=True).flatten()
        dfal_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./data_generate/svm_result/"+str(dataset)+"/feature/"+str(length)+"/dfalk/" + str(i) + ".npy", allow_pickle=True).flatten()
        dfal_k_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./data_generate/svm_result/"+str(dataset)+"/feature/"+str(length)+"/knockoff/" + str(i) + ".npy", allow_pickle=True).flatten()
        knockoff.append(temp)


    kcenter_list = random_reverse_array(np.array(kcenter_list))
    uncertain_list = random_reverse_array(np.array(uncertain_list))
    random_list = random_reverse_array(np.array(random_list))
    adfa_list = random_reverse_array(np.array(dfal_list))
    adfa_k_list = random_reverse_array(np.array(dfal_k_list))
    knockoff=random_reverse_array(np.array(knockoff))
    # trainx = np.concatenate([random_list[:30], uncertain_list[:30], kcenter_list[:30], adfa_list[:30], adfa_k_list[:30]])  #
    malicious = np.concatenate([kcenter_list,uncertain_list,  adfa_list, adfa_k_list,knockoff]) #
    benign = random_list

    return malicious, benign

def shang(probs):
    log_probs = np.log2(probs)
    shang = -1 * np.sum(probs * log_probs, axis=1)
    return shang

def acc(b_ac,m_ac):
    return b_ac*0.4+m_ac*0.6

def entropy1(data):    #返回每个样本的指数
    n=np.shape(data)[0]
    sumzb=np.sum(data,axis=0)
    data=data/sumzb
    #对ln0处理
    a=data*1.0
    a[np.where(data==0)]=0.0001
#    #计算每个指标的熵
    e=(-1.0/np.log(n))*np.sum(data*np.log(a),axis=0)
#    #计算权重
    w=(1-e)/np.sum(1-e)
    w = [round(i,4) for i in w]
    #print(w)
    recodes=np.sum(data*w,axis=1)
    return recodes

def calc_similarity(arr1, arr2):
    if len(arr1) != len(arr2):
        return "Error: The two arrays have different lengths!"

    count = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            count += 1

    similarity = count / len(arr1)
    return similarity
dataset="mnist"
length = 1800
data_ml,data_bn=my_read_data(dataset,length)

data_ml=data_ml

tr_y1=[[0]]*200
tr_y2=[[1]]*200

tr_y=np.vstack([tr_y1,tr_y2])#tr_y4,tr_y5

kcenter_train_x,kcenter_train_y,kcenter_test_x,kcenter_test_y=clip(data_ml[0:40],[1]*40)
uncertainty_train_x,uncertainty_train_y,uncertainty_test_x,uncertainty_test_y=clip(data_ml[40:80],[1]*40)
dfal_train_x,dfal_train_y,dfal_test_x,dfal_test_y=clip(data_ml[80:120],[1]*40)
dfalk_train_x,dfalk_train_y,dfalk_test_x,dfalk_test_y=clip(data_ml[120:160],[1]*40)
knockoff_train_x,knockoff_train_y,knockoff_test_x,knockoff_test_y=clip(data_ml[160:200],[1]*40)
b_train_x,b_train_y,b_test_x,b_test_y=clip(data_bn[:200],[0]*200)

X_train =np.concatenate([kcenter_train_x,uncertainty_train_x,dfal_train_x,dfalk_train_x,knockoff_train_x,b_train_x])
y_train = np.concatenate([kcenter_train_y,uncertainty_train_y,dfal_train_y,dfalk_train_y,knockoff_train_y,b_train_y])
# 创建一个SVM分类器并进行预测
clf = SVC(kernel='poly',C=1,degree=1,coef0=2,cache_size=1000,random_state=60)#创建SVM训练模型
# clf=SVC()
clf.fit(X_train,y_train)#对训练集数据进行训练
print("##benign##")
b_predict=clf.predict(b_test_x)
benign_scores = calc_similarity(b_predict,b_test_y)
print(benign_scores)

print("##kcenter##")
kcenter_predict=clf.predict(kcenter_test_x)
kcenter_scores = calc_similarity(kcenter_predict,kcenter_test_y)
print("acc:{}|frp:{}.".format(acc(benign_scores,kcenter_scores),(1-kcenter_scores)))

print("##uncertainty##")
uncertainty_predict=clf.predict(uncertainty_test_x)
uncertainty_scores = calc_similarity(uncertainty_predict,uncertainty_test_y)
print("acc:{}|frp:{}.".format(acc(benign_scores,uncertainty_scores),(1-uncertainty_scores)))

print("##dfal##")
dfal_predict=clf.predict(dfal_test_x)
dfal_scores = calc_similarity(dfal_predict,dfal_test_y)
print("acc:{}|frp:{}.".format(acc(benign_scores,dfal_scores),(1-dfal_scores)))

print("##dfalk##")
dfalk_predict=clf.predict(dfalk_test_x)
dfalk_scores = calc_similarity(dfalk_predict,dfalk_test_y)
print("acc:{}|frp:{}.".format(acc(benign_scores,dfalk_scores),(1-dfalk_scores)))

print("##knockoff##")
knockoff_predict=clf.predict(knockoff_test_x)
knockoff_scores = calc_similarity(knockoff_predict,knockoff_test_y)
print("acc:{}|frp:{}.".format(acc(benign_scores,knockoff_scores),(1-knockoff_scores)))

