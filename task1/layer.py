"""由层计算导数，仅支持链式"""
import numpy as np


class Linear:
    """线性层，可以多批量:input_shape=(batch,1,input)"""
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print("linear层init完成")
   

    def init_param(self,loc=0,scale=0.5)->None:
        """
        args:
            loc:int, weight均值
            scale:int, weight方差
        others:
            weight:shape=(1,inp,outp)
            bias:shape=(1,1,outp)
        """
        self.weight = np.random.normal(size=(1, self.num_input, self.num_output),
                                       loc=loc,scale=scale).astype(np.float32)
        self.bias = np.zeros(shape=(1,1,self.num_output),dtype=np.float32)
    
    def load_param(self, weight, bias)->None:
        """参数加载"""
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    
    def save_param(self)->None:
        """参数保存"""
        return self.weight, self.bias


    def forward(self,input):
        """
        arg:
            input:ndarray, shape=(batch,1,input_len)
        return:
            output:ndarray, shape=(batch,1,output_len)
        """
        assert len(input.shape) == 3, f"输入形状不对:{input.shape}"
        assert input.shape[2]==self.num_input, f"输入向量大小为{input.shape[2]}，不等于要求大小"
        assert hasattr(self, 'weight'), "没有初始化linear"
        self.input = input
        self.output = self.input @ self.weight + self.bias
        return self.output
    
    def backward(self, top_diff):
        """
        args:
            top_diff:ndarray, shape=output.shape
        return:
            bottom_diff:ndarray, shape=input.shape
        """
        assert top_diff.shape == self.output.shape
        self.d_weight = self.input.transpose(0,2,1)@top_diff
        self.d_bias = top_diff
        bottom_diff = top_diff@self.weight.transpose(0,2,1)
        return bottom_diff
    
    def update_param(self, lr):
        self.weight = self.weight - lr*np.sum(self.d_weight,axis=0,keepdims=True)
        self.bias = self.bias - lr*np.sum(self.d_bias,axis=0,keepdims=True)

    
class SoftmaxAndCrossEntropy():
    def __init__(self):
        print('Softmax层init完成')
        #执行完forward标记False，执行完get_loss标记True，防止标签、数据不匹配
        self.__backward_allow = False 

    def forward(self, input):
        """
        args:
            input:ndarray, shape=(batch,1,input_len)
        return:
            prob:ndarray, shape=(batch,1,input_len), 仅将input概率化并返回
        """
        assert len(input.shape)==3 and input.shape[1]==1 ,\
              f"softmax输入错误, input_shape:{input.shape}"
        input_max = np.max(input, axis=2, keepdims=True) #shape(batch,1,1)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp,axis=2,keepdims=True)
        self.__backward_allow = False
        return self.prob
    
    def backward(self):
        """
        args: None
        return: 
            bottom_diff:ndarray, shape=self.prob.shape
        """
        assert self.__backward_allow == True
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

    def get_loss(self, label):
        """
        args:
            label: ndarray, shape=(batchsize), dtype=int32, 用0起始的数字表示类别
        return:
            loss: ndarray, shape=(batch,1,1), 是CrossEntropyloss
        others:
            label_onehot: shape=self.prob.shape=(batch,1,input_len)
            执行这一步才能准备好backward的需求
        """
        assert len(label.shape)==1 and label.shape[0]==self.prob.shape[0]
        assert label.dtype == np.int32

        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob,dtype=np.float32)
        self.label_onehot[np.arange(self.batch_size), 0, label] = 1.0
        loss = -np.sum( self.label_onehot * np.log(self.prob+1e-12) ,axis=2) / self.batch_size

        self.__backward_allow = True
        return loss
    
    