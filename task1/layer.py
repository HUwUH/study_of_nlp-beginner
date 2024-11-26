"""由层计算导数，仅支持链式"""
import numpy as np


class Linear:
    """线性层，需要多批量:input_shape=(batch,input)"""
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print("linear层init完成")
    def init_param(self,loc=0,scale=0.1):
        """loc：均值，scale：方差.
            weight:shape=(1,inp,outp) bias:shape=(1,1,outp)
        """
        self.weight = np.random.normal(size=(1, self.num_input, self.num_output),
                                       loc=loc,scale=scale).astype(np.float32)
        self.bias = np.zeros(shape=(1,1,self.num_output),dtype=np.float32)
   
    def forward(self,input):
        """input:shape=(batch,input_len)"""
        assert len(input.shape) == 2
        assert input.shape[1]==self.num_input
        assert hasattr(self, 'weight')
        self.input = input.reshape(input.shape[0],1,-1) #修改形状为(batch,1,input)
        self.output = self.input @ self.weight + self.bias
        return self.output
    def backward(self, top_diff):
        assert top_diff.shape == self.output.shape
        self.d_weight = np.dot(self.input.transpose(0,2,1), top_diff)
        self.d_bias = top_diff
        bottom_diff = np.dot(top_diff,self.weight.transpose(0,2,1))
        return bottom_diff
    def update_param(self, lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias
    
    def load_param(self, weight, bias):
        """参数加载"""
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):
        """参数保存"""
        return self.weight, self.bias

class SoftmaxAndCrossEntropy():
    def __init__(self):
        print('Softmax层init完成')
    def forward(self, input):
        """input:shape=(batch,1,input_len)
            输出：softmax结束，格式(batch,1,input_len)
            inputlen 应当对应几种类别
        """
        assert len(input.shape)==3 and input.shape[1]==1
        input_max = np.max(input, axis=2, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp,axis=2,keepdims=True)
        return self.prob
    def get_loss(self, label):
        """lable为整数数组（1维向量），或者onehot (batch,1,input_len)"""
        self.batch_size = self.prob.shape[0]
        if label.ndim == 1:
            self.label_onehot = np.zeros_like(self.prob)
            self.label_onehot[np.arange(self.batch_size), 0, label] = 1.0
        elif label.shape == self.prob.shape:
            self.label_onehot = label
        else:
            raise ValueError("形状错误")
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), 0, label] = 1.0
        loss = -np.sum(np.log(self.prob + 1e-12) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff