import numpy as np

#1
def shuffle_and_split(content:list,train_rate:float):
    """准备训练、测试集
        content为二维数组
    """
    assert len(content)>10
    assert 0 < train_rate < 1
    sentences_num = len(content)
    train_num = int(sentences_num*train_rate)

    array = np.arange(sentences_num)
    np.random.shuffle(array)
    train_set = [content[i] for i in array[:train_num]]
    valid_set = [content[i] for i in array[train_num:]]

    return train_set,valid_set

#2
class Bag:
    def __init__(self,sentences):
        """sentences为一维数组，只存句子"""
        assert isinstance(sentences, list)
        self.data = sentences
        self.word2num = {'<unk>':0}
        self.dictlen = 1

    def init_vocab(self)->None:
        """初始化词表"""
        for sentence in self.data:
            sentence = sentence.lower()
            words = sentence.split()
            for word in words:
                if word not in self.word2num:
                    self.word2num[word] = len(self.word2num)
        self.dictlen = len(self.word2num)

    def get_vocab_len(self)->int:
        """词表长度"""
        return self.dictlen
    
    def trans_to_tensor(self,sentences):
        """将句子列表转换为张量表示
            sentences: [str]：存句子的列表
            输出：[[]]二维numpy列表，batchsize*dictlen
        """
        assert isinstance(sentences, list)
        batch_size = len(sentences)
        tensor = np.zeros((batch_size,self.dictlen),dtype=np.float32)

        for i,sentence in enumerate(sentences):
            sentence = sentence.lower()
            words = sentence.split()
            for word in words:
                idx = self.word2num.get(word, self.word2num['<unk>'])
                tensor[i, idx] += 1
        
        return tensor

class dataloader:
    def __init__(self,batchsize:int ,vocab ,data ,mode='train'):
        """
        data:  sentence_id:any,id:any,sentence:str(,lable:int)
        1.不负责shuffle
        2.对于batchsize剩下的舍弃
        """
        self.batchsize = batchsize
        self.vocab =vocab
        self.mode = mode

        self.sentences = [d[2] for d in data]
        self.labels = None
        self.num_batchs = len(self.sentences) // self.batchsize

        if mode == 'train':
            assert len(data[0]) == 4
            self.labels = [d[3] for d in data]
        elif mode == 'test':
            assert len(data[0]) >= 3
            self.labels = None
        else:
            raise ValueError

    def __len__(self):
        """batch长度"""
        return self.num_batchs
    
    def __getitem__(self,idx):
        """
        只支持索引访问
        test: 一个tensor（batchsize*vocablen)
        train: tensor（batchsize*vocablen), list长batchsize存类别的数字        
        """
        if idx >= self.num_batchs or idx < 0:
            raise IndexError("Batch index out of range.")
        #获取切片
        begidx = idx*self.batchsize
        endidx = (idx+1)*self.batchsize
        out_sentences = self.sentences[begidx:endidx]
        #转成tensor
        out_tensor = self.vocab.trans_to_tensor(out_sentences)
        if self.mode == 'test':
            return out_tensor
        elif self.mode == 'train':
            out_label = self.labels[begidx:endidx]
            return out_tensor, out_label

    

if __name__ == "__main__":
    def test_shuffle():
        sentences = [[1*i,2*i,3*i] for i in range(20)]
        ts,vs = shuffle_and_split(sentences,0.5)
        print(ts,vs)
    test_shuffle()
