from cnn_text.instance import Instance,AlphaBet,Example
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from cnn_text.model import CNN_text
from cnn_text.Hyperparam import Parameter

class Mydata:
    def seperate(self,path):
        f=open(path,'r')
        dataset=[]
        for line in f.readlines():          #把数据集合按每句分割
            instance=Instance()
            info=line.strip().split("|||")      #strip()用于移除头和尾部空格，split()分割词和标签
            instance.words=info[0].split(' ')    #把分割后的每个词用空格分开，传入一个集合中words标签
            instance.labels=info[1].strip()
            #print('words:',instance.words)
            #print('label',instance.labels)
            dataset.append(instance)
        f.close()
        return dataset

    def __init__(self):
        self.parameter=Parameter()
    def createAlphaBet(self,dataset):       #dataset是instance中得到
        allwords=[]
        allwords.append('unknow')
        alllabels=[]
        for ds in dataset:                 #遍历得到的数据集的每一句话
            for i in ds.words:              #遍历每一句话的每一个单词
                allwords.append(i)
            for j in ds.labels:             #遍历每一句话的每一个标签
                alllabels.append(j)
        alpha=AlphaBet(allwords)        #此时alpha中存放的是去重之后的word
        bet=AlphaBet(alllabels)         #bet中存放的是去重之后的label

        self.parameter.embed_num=len(alpha.word_dic)    #这里可以得到embed_num是50维，也是我们设置参数自己设置初始值
        self.parameter.label_size=len(bet.word_list)    #这里是五分类，得到label_size是label的值

        return alpha,bet

    def changeWords_num(self,w_index,l_index,receiveResult):# 把seperate结果集传到receiveResult

        allexample=[]       #allexample表示一串example的集合，而example,存放的是allwords_index，alllabels_index
        for i in receiveResult:
            example=Example()
            for x in i.words:
                if x in w_index.word_dic:
                    example.allwords_index.append(w_index.word_dic[x]) #通过单词下标查字典，把查找的结果封装到example对象的属性中
                else:
                    example.allwords_index.append(w_index.word_dic['unknow'])
            example.alllabels_index=l_index.word_dic[i.labels]     #通过label下标查字典，赋值给example中的存放的alllabels_index
            #return example.allwords_index,example.alllabels_index
            allexample.append(example)
        return allexample

    def toVariable(self,example):       #把存到词的下标和标签下标转为向量与torch相接
        x=torch.autograd.Variable(torch.LongTensor(1,len(example.allwords_index)))
        y=torch.autograd.Variable(torch.LongTensor(1))
        for idx in range(len(example.allwords_index)):
            x.data[0][idx]=example.allwords_index[idx]
        y.data[0]=example.alllabels_index
        #print('x:',x)       #x表示的是1*x维一句话中单词的向量
        #print('y:',y)       #y表示的是1*y维的没一句话每个标签的向量
        return x,y

    def train(self,train_path,dev_path,test_path):
        trainRes=self.seperate(train_path)
        devRes=self.seperate(dev_path)
        testRes=self.seperate(test_path)
        wordAlpha,labelBet=self.createAlphaBet(trainRes)#得到训练集去重的词典
        trainSet=self.changeWords_num(wordAlpha,labelBet,receiveResult=trainRes)#转化为数字
        devSet=self.changeWords_num(wordAlpha,labelBet,receiveResult=devRes)
        testSet=self.changeWords_num(wordAlpha,labelBet,receiveResult=testRes)

        #添加优化器
        self.model=CNN_text(self.parameter)
        optimizer=torch.optim.Adagrad(self.model.parameters(),lr=self.parameter.learnRate)
        total_num=len(trainSet)
        print('trainSet=',total_num)
        for n in range(1,100):
            print('第%d次循环:'% n)
            sum=0
            correct=0
            for i in trainSet:
                optimizer.zero_grad()
                x, y = self.toVariable(i)
                logit = self.model.forward(x)
                loss = F.cross_entropy(logit, y)    #目标函数求导
                loss.backward()                     #方向传播
                #print('loss:', loss)
                optimizer.step()
                if y.data[0]==self.getMaxIndex(logit):
                    correct+=1
                sum+=1
            print('acc=',correct / sum)
        return trainSet,devSet,testSet

    def getMaxIndex(self,score):    #获取最大权重的下标
        label_size=score.size()[1]
        maxIndex=0
        max=score.data[0][0]
        for idx in range(label_size):
            tmp=score.data[0][idx]
            if max<tmp:
                max=tmp
                maxIndex=idx
        return maxIndex






mydata=Mydata()
path1='D:/python/cnn_text/data/raw.clean.train'
path2='D:/python/cnn_text/data/raw.clean.dev'
path3='D:/python/cnn_text/data/raw.clean.test'
mydata.train(path1,path2,path3)
#res=mydata.seperate(path)
#wordAlpha,labelBet=mydata.createAlphaBet(res)
#mydata.changeWords_num(wordAlpha,labelBet)

#mydata.toVariable()
