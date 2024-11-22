"""由层计算导数，仅支持链式"""
import numpy as np

class Linear:
    """线性层，允许多批量"""
    def __init__(self,input_length,output_length):
        self.input_length = input_length
        self.output_length = output_length
        print("linear层init完成")

    def init_param(self,wloc=1,wscale=1):
        """loc：均值，scale：方差"""
        self.weight = np.random.normal(size=[self.input_length,self.output_length],
                                       loc=wloc,scale=wscale).astype(np.float32)
        self.bias = np.random.normal(size=[self.output_length],
                                     loc=0,scale=1).astype(np.float32)
    
    def forward(self,input):
        assert len(input.shape) == 2
        assert input.shape[1]==self.input_length
        self.input = input
        self.
        
class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    def forward(self, input):  # 前向传播计算
        start_time = time.time()
        self.input = input
        # TODO：全连接层的前向传播，计算输出结果
        self.output = np.dot(self.input, self.weight)+self.bias
        return self.output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        self.d_weight = np.dot(top_diff, self.input.T)
        self.d_bias = top_diff
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff
    def update_param(self, lr):  # 参数更新
        # TODO：对全连接层参数利用参数进行更新
        self.weight = self.weight-lr*self.d_weight
        self.bias = self.bias-lr*self.d_bias
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):  # 参数保存
        return self.weight, self.bias