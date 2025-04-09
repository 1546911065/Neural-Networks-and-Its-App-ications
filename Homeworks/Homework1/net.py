import torch.nn as nn # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块

# 定义波士顿房价预测的神经网络模型
class housing_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.Linear(13,128)   # 定义第一层全连接层，输入维度为13（波士顿房价数据的特征数），输出维度为128
        self.hidden2=nn.Linear(128,256)  # 定义第二层全连接层，输入维度为128，输出维度为256
        self.hidden3=nn.Linear(256,256)  # 定义第三层全连接层，输入维度为256，输出维度为256
        self.out=nn.Linear(256,1)        # 定义输出层，输入维度为256，输出维度为1（房价预测值）
        self.drop=nn.Dropout(0.05)       # 定义Dropout层，用于防止过拟合，丢弃概率为5%
    def forward(self,x):
        x=F.relu(self.hidden1(x))        # 第一层：线性变换 + ReLU激活函数 + Dropout
        x=self.drop(x)
        x=F.relu(self.hidden2(x))        # 第二层：线性变换 + ReLU激活函数 + Dropout
        x=self.drop(x)
        x=F.relu(self.hidden3(x))        # 第三层：线性变换 + ReLU激活函数 + Dropout
        x=self.drop(x)
        x=self.out(x)
        x=x.squeeze(-1)                  # 移除输出张量的最后一个维度（将形状从[N, 1]变为[N]）
        return x
