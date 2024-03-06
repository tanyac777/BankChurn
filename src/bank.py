from pandas import read_csv
from sklearn import datasets

f=open("data/train.csv",encoding='UTF-8')
names=['id','CustomerId','Surname','CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']
next(f)
data=read_csv(f,names=names)
print(data)

X = data.iloc[:,1:-1]
Y = data.iloc[:,-1]
print(X)
print(Y)
    
