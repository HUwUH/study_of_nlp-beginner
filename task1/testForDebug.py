"""
我自己的代码（bow）的正确率仅仅为25%不到，难以在训练集合收敛。
因此，尝试结合reference中其他人写的的BoW和我的模型融合，试试是不是我模型错了
这个代码能训练到70%以上，应该不是我的模型的问题，而是我的BoW可能有问题。
我猜测，可能是我的BoW的词表长度太长导致的。

    修改了我的BoW为词表大小可改动，但是1.选取全部句子，准确率50%左右，2.只选取全
句，不理想，准确率仅仅30%

    发现了，他的data_split没有shuffle，也没有截取仅仅完整句子，相当于句子的数目
不仅小了很多，而且整句的数量少了非常多，仅仅36句，且测试集和训练集有相同句子段。
我的代码加上这种调整也能到67%以上！他的代码不适用这种做法也是50%左右的准确率。
    没有使用他的代码测试“只训练、测试整句而不考虑句子片段”的情况
"""

import numpy
import random
import layer
import csv

def data_split(data, test_rate=0.3, max_item=1000):
    """把数据按一定比例划分成训练集和测试集"""
    train = list()
    test = list()
    i = 0
    for datum in data:
        i += 1
        if random.random() > test_rate:
            train.append(datum)
        else:
            test.append(datum)
        if i > max_item:
            break
    return train, test


class Bag:
    """Bag of words"""
    def __init__(self, my_data, max_item=1000):
        self.data = my_data[:max_item]
        self.max_item=max_item
        self.dict_words = dict()  # 单词到单词编号的映射
        self.len = 0  # 记录有几个单词
        self.train, self.test = data_split(my_data, test_rate=0.3, max_item=max_item)
        self.train_y = [int(term[3]) for term in self.train]  # 训练集类别
        self.test_y = [int(term[3]) for term in self.test]  # 测试集类别
        self.train_matrix = None  # 训练集的0-1矩阵（每行一个句子）
        self.test_matrix = None  # 测试集的0-1矩阵（每行一个句子）

    def get_words(self):
        for term in self.data:
            s = term[2]
            s = s.upper()  # 记得要全部转化为大写！！（或者全部小写，否则一个单词例如i，I会识别成不同的两个单词）
            words = s.split()
            for word in words:  # 一个一个单词寻找
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.test_matrix = numpy.zeros((len(self.test), self.len))  # 初始化0-1矩阵
        self.train_matrix = numpy.zeros((len(self.train), self.len))  # 初始化0-1矩阵

    def get_matrix(self):
        for i in range(len(self.train)):  # 训练集矩阵
            s = self.train[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.train_matrix[i][self.dict_words[word]] = 1
        for i in range(len(self.test)):  # 测试集矩阵
            s = self.test[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.test_matrix[i][self.dict_words[word]] = 1


#数据读取
with open(r'.\task1\data\train.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)

# 初始化
data = temp[1:]
max_item=1000
random.seed(42)
numpy.random.seed(42)

# 特征提取
bag=Bag(data,max_item)
bag.get_words()
bag.get_matrix()

print(bag.len) #363

class MyLinearModel:
    """耦合度较高，必须先forward，再getloss，再backward"""
    def __init__(self,input,output,lr):
        self.linear = layer.Linear(input,output)
        self.softmax = layer.SoftmaxAndCrossEntropy()
        self.input_len = input
        self.last_input_batchsize = None
        self.learningrate = lr

        self.linear.init_param()

    def forward(self,x):
        """返回softmax以后的"""
        assert isinstance(x,numpy.ndarray)
        self.last_input_batchsize = x.shape[0]
        x = self.linear.forward(x)
        x = self.softmax.forward(x)
        return x
    def getloss(self,label):
        """返回loss"""
        assert label.shape[0] == self.last_input_batchsize
        return self.softmax.get_loss(label)
    def backward(self):
        mid_stream = self.softmax.backward()
        mid_stream = self.linear.backward(mid_stream)
        self.linear.update_param(self.learningrate)

lr = 0.01
model = MyLinearModel(bag.len,5,lr)

epoch = 3000
for ep in range(epoch):
    input_tensor,lable = bag.train_matrix,bag.train_y
    lable = numpy.array(lable,dtype=numpy.int32)
    soft_outp = model.forward(
        input_tensor.reshape(input_tensor.shape[0],1,-1))
    model.getloss(lable)
    model.backward()
    # if ep%10 == 0:
    #     soft_outp = soft_outp.reshape(-1,soft_outp.shape[-1])
    #     print("batch:",ep)
    #     print("softmax的结果:",soft_outp[:5],sep='\n')
    #     print("lable:",lable[:5])
        
    #     ans = numpy.argmax(soft_outp,axis=1)

    #     right = numpy.sum(ans==lable)
    #     print("accuracy：",right/ans.shape[0])
    if ep%100 == 0:
        input_tensor,lable = bag.test_matrix,bag.test_y
        lable = numpy.array(lable,dtype=numpy.int32)
        soft_outp = model.forward(
            input_tensor.reshape(input_tensor.shape[0],1,-1))
        soft_outp = soft_outp.reshape(-1,soft_outp.shape[-1])
        print("batch:",ep)
        print("softmax的结果:",soft_outp[:5],sep='\n')
        print("lable:",lable[:5])
        
        ans = numpy.argmax(soft_outp,axis=1)

        right = numpy.sum(ans==lable)
        print("accuracy：",right/ans.shape[0])
        
