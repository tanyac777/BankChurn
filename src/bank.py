import pandas as pd
from pandas import read_csv
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
# 读取数据,选取数据进行建模，去除CustomerId和Surname等标记特征
f=open("data/train.csv",encoding='UTF-8')
# 合并原始数据集
f_origin=open("data/Churn_Modelling.csv",encoding='UTF-8')

names=['id','CustomerId','Surname','CreditScore','Geography',
       'Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard',
       'IsActiveMember','EstimatedSalary','Exited']
next(f)
next(f_origin)
data=read_csv(f,names=names)
data_origin=read_csv(f_origin,names=names)
data_merge=pd.concat([data,data_origin],ignore_index=True)



# 选取数据范围
X = data_merge.iloc[:,3:-1]
y = data_merge.iloc[:,-1]

# 将属性转为数字标识
from sklearn import preprocessing
Xdf = pd.DataFrame(X)
le = preprocessing.LabelEncoder()
for col in Xdf.columns[1:3]:
    f = le.fit_transform(Xdf[col])
    Xdf[col] = f
print(Xdf)

# 处理缺失值
Xdf.info()
# Geography缺失1，Age缺失1，HasCrCard缺失1，IsActiveMember缺失1
Geography = Xdf.loc[:,"Geography"].values.reshape(-1,1)
Age = Xdf.loc[:,"Age"].values.reshape(-1,1)
HasCrCard = Xdf.loc[:,"HasCrCard"].values.reshape(-1,1)
IsActiveMember = Xdf.loc[:,"IsActiveMember"].values.reshape(-1,1)
from sklearn.impute import SimpleImputer

# 用中位数填补原数据
imp_median = SimpleImputer(strategy="median")
Xdf.loc[:,"Geography"] = imp_median.fit_transform(Geography)
Xdf.loc[:,"Age"] = imp_median.fit_transform(Age)
Xdf.loc[:,"HasCrCard"] = imp_median.fit_transform(HasCrCard)
Xdf.loc[:,"IsActiveMember"] = imp_median.fit_transform(IsActiveMember)
Xdf.info()

# 对编码后的数字进行独热编码
#enc = preprocessing.OneHotEncoder()
#Xdf_enc = enc.fit_transform(Xdf)
#print(Xdf_enc)


# 设置训练数据集和测试数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

# 数据标准化
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
# 将训练数据标准化
X_train_std = stdsc.fit_transform(X_train)
# 将测试数据标准化
X_test_std = stdsc.transform(X_test)


# 类似地，对测试集进行预处理
f_t=open("data/test.csv",encoding='UTF-8')
next(f_t)
data_t=read_csv(f_t,names=names)
X_t = data_t.iloc[:,3:-1]
Xdf_t = pd.DataFrame(X_t)
le = preprocessing.LabelEncoder()
for col in Xdf_t.columns[1:3]:
    f_t = le.fit_transform(Xdf_t[col])
    Xdf_t[col] = f_t
print(Xdf_t)
X_t_std = stdsc.transform(X_t)
print(X_t_std)

# 逻辑回归方法
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=10)
# lr在原始测试集上的表现
lr.fit(X_train_std, y_train)
# 打印训练集精确度
print('Training accuracy:', lr.score(X_train_std, y_train))
# 打印测试集精确度
print('Test accuracy:', lr.score(X_test_std, y_test))

# 打印系数
print(lr.coef_)
# 打印截距
print(lr.intercept_)

# 模型评价
# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
y_pred = lr.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

# 将混淆矩阵可视化
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

#plt.xlabel('prediction')
#plt.ylabel('reality')
#plt.show()

# 获取模型的准确率和召回率
from sklearn.metrics import precision_score, recall_score, f1_score
# 准确率
print('Precision: %.4f' % precision_score(y_true=y_test, y_pred=y_pred))
# 召回率
print('Recall: %.4f' % recall_score(y_true=y_test, y_pred=y_pred))
# F1
print('F1: %.4f' % f1_score(y_true=y_test, y_pred=y_pred))

from sklearn.metrics import roc_curve, auc
from scipy import interp


# 设置图形大小
fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
# 计算 预测率
probas = lr.fit(X_train, y_train).predict_proba(X_test)
# 计算 fpr,tpr    
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label=1)
mean_tpr += interp(mean_fpr, fpr,  tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' 
                    % ( roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

mean_tpr /= len(X_train)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('')
plt.legend(loc="lower right")
plt.show()

# 在测试集上预测
list=lr.predict_proba(X_t_std)
list1=[]
for i in range(0,len(list)):
	list1.append(list[i][1])
name=['Exited']
test=pd.DataFrame(columns=name,data=list1)
print(test)
test.to_csv('data/result.csv',encoding='gbk')