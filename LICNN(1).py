##实现李想算例————卷积神经网络求解线性方程组。
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,train_test_split
import pandas as pd
import numpy as np
import math

# Hyper parameters
EPOCH = 25
BATCH_SIZE = 1
LR = 0.1
num = 400   #方程组维数
# 载入数据
zscore = sklearn.preprocessing.StandardScaler()  #用作归一化
df = pd.read_csv(r"D:\python_work\刚度矩阵.csv") #shape:(400,401)
X = df.values[0:num, 0:num]  #二维数组
# X = zscore.fit_transform(X)
X = np.expand_dims(X, axis=0)  # 增加一维轴,变为（1,400,400）
X = np.expand_dims(X, axis=0)  # 增加一维轴,变为（1,1,400,400）
Y = df.values[0:num, num]  #shape:(400,)
Y = np.expand_dims(Y, axis=0)
# Y = zscore.fit_transform(Y)
# Y = np.ravel(Y)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(  # --> (1,400,400)
                in_channels=1,  # 传入的数据（图片）是几层的，灰色为1层，RGB为三层
                out_channels=1,  # 输出的图片是几层
                kernel_size=(1,num),  # 代表扫描的区域点为1*6
                stride=1,  # 就是每隔多少步跳一下
                bias=False, #偏置必须设置为0
                padding=0,  # 边框补全，其计算公式=（kernel_size-1）/2=
            )  # 2d代表二维卷积

    def forward(self, x):
        y1 = self.c1(x)
        y1 = y1.reshape((-1)) #改变形状 拉平
        return y1

cnn = CNN()

# 添加优化方法
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 指定损失函数使用MSE
loss_fn = nn.MSELoss()

##-----------开始训练--------------
step = 0
for epoch in range(EPOCH):
    # 加载训练数据
    for step in range(150):
        # 分别得到训练数据的x和y的取值
        X = torch.tensor(X)
        Y = torch.tensor(Y) #转化为张量格式
        b_x = Variable(X)
        b_y = Variable(Y)
        b_x = torch.tensor(b_x, dtype=torch.float32) #转化为double格式
        b_y = torch.tensor(b_y, dtype=torch.float32) # 转化为double格式
        b_y = b_y.reshape((-1))
        output = cnn(b_x)  # 调用模型预测
        loss = loss_fn(output, b_y)  # 计算损失值
        accuracy = (sum((b_y - output<0.001)).item()) / 400.0
        print('now epoch :  ', epoch, '   |  loss : %.4f ' % loss.item(), '     |   accuracy :   ', accuracy)
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降
        optimizer.zero_grad()  # 每一次循环之前，将梯度清零


for name in cnn.state_dict():
   print(name)
#print(cnn.state_dict()['c1.weight'])
a=cnn.state_dict()['c1.weight'] #四维张量
a = a.squeeze(0)
a = a.squeeze(0)#转为二维张量
b = a.numpy()#把二维张量转化为二维数组
# b = zscore.inverse_transform(b)
b = np.ravel(b)
print('方程组的解（卷积层权重）=',b)
np.savetxt("D:\python_work\solutions.csv",b) #将最终方程组的解保存到CSV文件中

#计算与精确解的误差
erro = 0
Z = df.values[0:num, num+1] #读取精确解
for i in range(num):
    erro = erro + (math.fabs(Z[i] - b[i]))

erro = erro / 400.0 #平均相对误差
print('与精确解的平均相对误差=',erro)