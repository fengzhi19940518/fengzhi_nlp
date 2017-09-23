import torch.nn as nn
import torch.nn.functional as F

class CNN_text(nn.Module):
    def __init__(self,parameter):
        super(CNN_text,self).__init__()
        self.parameter=parameter
        self.hidden_size =parameter.hidden_size
        self.label_size = parameter.label_size
        self.embed_num = parameter.embed_num
        self.embed_dim = parameter.embed_dim
        self.class_num = parameter.class_num
        self.learnRate = parameter.learnRate

        self.embedding=nn.Embedding(parameter.embed_num,parameter.embed_dim)    #embedding的作用是把一个单词变成一个1*多维的向量

        print('embed_dim=%d,label_size=%d'%(self.embed_dim,self.label_size))

        self.fc1=nn.Linear(parameter.embed_dim,parameter.label_size)#通过linear函数把两矩阵进行合并需求矩阵，1*50与50*5向量相乘得到1*

    def forward(self,x):        #这里的x是toVariable中转化的x
        x=self.embedding(x)

        x=F.max_pool1d(x.permute(0,2,1),x.size()[1])        #max_poolld取一句话最大的特征值,进过调维度变成

        logit=self.fc1(x.view(1, self.parameter.embed_dim))
        return logit
