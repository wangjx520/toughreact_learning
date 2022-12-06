#静静手打版，复制应该没有问题吧
import os
import numpy as np
import csv

train=list(csv.reader(open('./train/train.csv','r',encoding='utf-8')))
trainx=[]
trainy=[]
for i in range(len(train)):
    number=i+1
    f='./train/'+str(number)+ '.csv'
    if not os.path.exists(f):
        continue

    x1=float(train[i][1])/0.3
    x2=float(train[i][2]) * 1e12
    x3=(float(train[i][3]) - 100) / 150.
    x4=(float(train[i][4]) - 25) / 25.
    l=[x1,x2,x3,x4,0]
    a=np.array(l).reshape((1,5))
    csvs =list(csv.reader(open(f,'r',encoding= 'utf-8')))

    count=np.ones(50)
    for index in range(1,len(csvs)):
        t=float(csvs[index][1])
        time=int(t-0.00001)
        xt=float(t)/50
        if (count[time]<=0):
            continue
        else:
            count[time]-=1
        vec = a.copy()
        vec[0][-1] = xt
        if(len(trainx)== 0):
            trainx = vec
        else:
            trainx=np.concatenate((trainx,vec),axis=0)
        y=np.zeros((1,1))
        y[0][0]=float(csvs[index][2])-189.
        if(len(trainy) == 0):
            trainy=y
        else:
            trainy = np.concatenate((trainy,y),axis=0)
#trainx = trainx.reshape(103, 50, 5)
#print(trainx, trainy)
print(trainx.shape,trainy.shape)
np.save('trainx',trainx)
np.save('trainy',trainy)




