class Instance:
    def __init__(self):
        self.words=[]
        self.labels=''

class AlphaBet:
    def __init__(self,dataset):     #dataset表示分割后的集合
        self.word_dic={}
        self.word_list=[]

        for word in dataset:
            if word not in self.word_list:      #这一步去除中重复的单词
                self.word_list.append(word)
        for i in range(len(self.word_list)):    #加入去重之后的词加入新建的词典中
            self.word_dic[self.word_list[i]]=i
        #print('dic:',self.word_dic)

class Example:                  #Example用来表示所有词组转换成数字的下标，而且每个数字对应字典的下标，其中包括单词下标和标签下标
    def __init__(self):
        self.allwords_index=[]
        self.alllabels_index=0      #初始化为0的原因是因为一句话只有一个标签，所以没有必要初始化为list集合
