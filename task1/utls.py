import numpy as np
from collections import Counter

def shuffle_and_split(content:list,first_rate:float):
    """
    分割数据集(列表)
    args:
        content: list
        first_rate: int, 分割出的两个中，前者的比例
    """
    assert len(content)>5
    assert 0 < first_rate < 1
    content_num = len(content)
    first_num = int(content_num*first_rate+0.5)

    array = np.arange(content_num)
    np.random.shuffle(array)
    first_set = [content[i] for i in array[:first_num]]
    second_set = [content[i] for i in array[first_num:]]

    return first_set,second_set


class Bag:
    def __init__(self, sentences, vocab_maxsize:int=-1):
        """
        args:
            sentences: list[str], 句子列表.
            vocab_maxsize: int。按从高到低频率最多存词语数量。-1不限大小
        others:
            vocab0为<unk>
        """
        assert isinstance(sentences, list)
        assert vocab_maxsize==-1 or vocab_maxsize>1
        self.data = sentences
        self.vocab_maxsize = vocab_maxsize
        self.word2num = {'<unk>':0}
        self.vocabsize = 1

        #初始化词表
        self.__init_vocab()

    def __init_vocab(self)->None:
        """初始化词表"""
        word_counter = Counter()

        #统计词频
        for sentence in self.data:
            words = sentence.lower().split()
            word_counter.update(words)
        
        #加入高频词
        most_common_words = word_counter.most_common(
            self.vocab_maxsize-1 if self.vocab_maxsize > 0 else None
            ) # -1因为已经有一个<unk>了
        for word, _ in most_common_words:
            if word not in self.word2num:
                self.word2num[word] = len(self.word2num)

        self.vocabsize = len(self.word2num)

    def get_vocab_size(self)->int:
        """词表长度"""
        return self.vocabsize
    
    def get_vocab(self)->dict:
        """返回词表字典"""
        return self.word2num

    def trans_to_tensor(self,sentences):
        """
        将句子列表转换为张量表示
        args:
            sentences: list[str], 存句子的列表
        return:
            tensor: np.ndarray, shape=(batch_size, vocab_size)
        """
        assert isinstance(sentences, list)
        batch_size = len(sentences)
        tensor = np.zeros((batch_size,self.vocabsize),dtype=np.float32)

        for i,sentence in enumerate(sentences):
            words = sentence.lower().split()
            for word in words:
                idx = self.word2num.get(word, self.word2num['<unk>'])
                tensor[i, idx] += 1
        
        return tensor


class Ngram:
    """和Bag基本差不多"""
    def __init__(self, sentences, maxN:int, gram_vocab_maxsize:int=-1):
        """
        args:
            sentences: list[str], 句子列表.
            maxN: int, N-gram中支持的最大N(包含)
            gram_vocab_maxsize: int。按从高到低频率最多存词语数量。-1不限大小。所有n
                分别排序, vocab总大小为gram_vocab_maxsize*maxN。
        others:
            不再包含<unk>
        """
        assert isinstance(sentences,list) and isinstance(sentences[0],str)
        assert gram_vocab_maxsize==-1 or gram_vocab_maxsize>0
        assert maxN<10, "N-gram的N不建议太大。真想要这样操作的话，就删了这行就行。"
        self.data = sentences
        self.vocab_maxsize = gram_vocab_maxsize
        self.word2num = {}
        self.vocabsize = 0
        self.maxN = maxN

        #初始化词表
        self.__init_vocab()

    def __init_vocab(self)->None:
        """初始化词表"""
        for n in range(1,self.maxN+1):
            word_counter = Counter()

            #统计词频
            for sentence in self.data:
                words = sentence.lower().split()
                nwords = self.__generate_ngrams(words, n)
                word_counter.update(nwords)
            
            #加入高频词
            most_common_words = word_counter.most_common(
                self.vocab_maxsize if self.vocab_maxsize > 0 else None
                )
            for word, _ in most_common_words:
                if word not in self.word2num:
                    self.word2num[word] = len(self.word2num)

        self.vocabsize = len(self.word2num)


    def get_vocab_size(self)->int:
        """词表长度"""
        return self.vocabsize

    def get_vocab(self)->dict:
        """返回词表字典"""
        return self.word2num

    def trans_to_tensor(self,sentences):
        """
        将句子列表转换为张量表示
        args:
            sentences: list[str], 存句子的列表
        return:
            tensor: np.ndarray, shape=(batch_size, vocab_size)
        """
        assert isinstance(sentences, list)
        batch_size = len(sentences)
        tensor = np.zeros((batch_size,self.vocabsize),dtype=np.float32)

        for i,sentence in enumerate(sentences):
            words = sentence.lower().split()
            for n in range(1, self.maxN + 1):
                ngrams = self.__generate_ngrams(words, n)
                for ngram in ngrams:
                    idx = self.word2num.get(ngram, None)
                    if idx is not None:
                        tensor[i, idx] += 1

        return tensor

    def __generate_ngrams(self,words, n):
        """生成词表的n长组合"""
        return ['_'.join(words[i:i+n]) for i in range(len(words)-n+1)]


class Dataloader:

    def __init__(self,batchsize:int,input,label=None,debug_info=None,input_transform=None,label_transform=None):
        """
        args:
            batchsize: int
            input: 可切片[beg:end]的对象，实现了len
            label: 非必须，可切片[beg:end]的对象，实现了len
            debug_info: 非必须，可切片[beg:end]的对象，实现了len
            input_transform: 非必须，函数列表或函数
            label_transform: 非必须，函数列表或函数
        others:
            1.不负责shuffle
            2.对于batchsize剩下的舍弃
            3.只支持索引访问
        """
        self.batchsize = batchsize
        self.num_batchs = len(input) // self.batchsize

        self.input = input
        self.label = label
        self.debuginfo = debug_info

        self.inp_trans = input_transform
        self.lab_trans = label_transform

        if label!=None:
            assert len(input)==len(label) 
        if debug_info!=None:
            assert len(input)==len(debug_info)

    def __len__(self):
        """batch长度"""
        return self.num_batchs
    
    def _batchsize(self):
        return self.batchsize

    def __getitem__(self,idx):
        """
        只支持索引访问
        args:
            idx: idx>=0 且 idx<num_batch
        return:
            input: input_transform(input[beg:end])
            lable: None, 或者label_transform(lable[beg:end])
            debuginfo: None,或者debug_info[beg:end]
        """
        #获取索引
        if idx >= self.num_batchs or idx < 0:
            raise IndexError("Batch index out of range.")
        begidx = idx*self.batchsize
        endidx = (idx+1)*self.batchsize
        
        #input
        input = self.input[begidx:endidx]
        if self.inp_trans!=None:
            if isinstance(self.inp_trans,list):
                for func in self.inp_trans:
                    input = func(input)
            else:
                input = self.inp_trans(input)
        #lable
        if self.label!=None:
            label = self.label[begidx:endidx]
            if self.lab_trans!=None:
                if isinstance(self.lab_trans,list):
                    for func in self.lab_trans:
                        label = func(label)
                else:
                    label = self.lab_trans(label)
        else:
            label = None
        #debug
        if self.debuginfo!=None:
            debuginfo = self.debuginfo[begidx:endidx]
        else:
            debuginfo = None
        
        return input, label, debuginfo

    

if __name__ == "__main__":
    import csv
    def test_shuffle():
        sentences = [[1*i,2*i,3*i] for i in range(20)]
        ts,vs = shuffle_and_split(sentences,0.5)
        print(ts,vs)
    test_shuffle()