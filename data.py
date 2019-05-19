import torch
import torchvision
import numpy as np
import copy
from dreamai.utils import *
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler,BatchSampler
import multiprocessing as mp,os
from torchvision import datasets, transforms, models
from dreamai.utils import *
import time
import string
import copy
import codecs
import unicodedata
from collections import Counter,defaultdict
import nltk
import pickle
from collections import Counter
torch.cuda.empty_cache()


class CharTextData():
    def __init__(self,text,vocab_size,batch_size=8,sequence_length=50,split_frac=0.1,valid=True,test_size=0.1):
        super().__init__()       

        self.vocab = tuple(set(text))
        self.text = text
        self.vocab_size = vocab_size
        self.int2char = dict(enumerate(self.vocab))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        #print(self.char2int)
        self.valid_text = None
        self.train_text,self.test_text = split_text(text,test_size=0.1)
        if valid:
            self.train_text,self.valid_text = split_text(self.train_text,test_size=test_size)
        else:
            self.valid_text = self.test_text

        self.train = self._make_batches(self.train_text)
        self.valid = self._make_batches(self.valid_text) 
        self.test = self._make_batches(self.test_text) 
        
        

    def get_train_size(self):
        return self.train['n_batches']
    
    def get_test_size(self):
        return self.test['n_batches']

    def get_valid_size(self):
        return self.valid['n_batches']
    
    def get_num_unique_chars(self):
        return len(set(self.text))
    
    def _make_batches(self,text):
        
        batch_size_total = self.batch_size * self.sequence_length
        batch_dict = {}
        batch_dict['encoded'] = np.array([self.char2int[ch] for ch in text])
        batch_dict['n_batches'] = len(batch_dict['encoded'])//batch_size_total
        batch_dict['batches'] = batch_dict['encoded'][:batch_dict['n_batches'] * batch_size_total].reshape((self.batch_size, -1))
        print(batch_dict['n_batches'])
        
        return batch_dict

    def get_batches(self,batches):
        
        # iterate through the array, one sequence at a time
        for n in range(0, batches.shape[1], self.sequence_length):
            # The features
            #print('n = {}'.format(n))
            x = batches[:, n:n+self.sequence_length]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], batches[:, n+self.sequence_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], batches[:, 0]
            yield torch.tensor(x),torch.tensor(y)

class TextLanguageModelData():
    def __init__(self,batch_size,is_valid=True,test_size=0.1):
        super().__init__()
        
        self.batch_size = batch_size
        
        self.valid = None
        
        self.train,self.test = split_text(self.tokens,test_size=test_size)
        if is_valid:
            self.train,self.valid = split_text(self.train,test_size=test_size)
        else:
            self.valid = self.test
             

        print('making batches')
        self.num_train_batches,self.train_batches = make_batches(self.train,
                                                                 self.batch_size,
                                                                 self.sequence_length,
                                                                 self.encoding_dict,
                                                                 self.num_cores)
        print('number of train batches = {}'.format(self.num_train_batches))
           
        self.num_test_batches,self.test_batches = make_batches(self.test,
                                                               self.batch_size,
                                                               self.sequence_length,
                                                               self.encoding_dict,
                                                               self.num_cores)
        print('number of test batches = {}'.format(self.num_test_batches))

        if is_valid:
            self.num_valid_batches,self.valid_batches = make_batches(self.valid,
                                                                     self.batch_size,
                                                                     self.sequence_length,
                                                                     self.encoding_dict,
                                                                     self.num_cores)
        else:
            self.valid_batches = self.test_batches
        print('number of valid batches = {}'.format(self.num_valid_batches))


        
        
       
        print('train batches shape = {}'.format(self.train_batches.shape))
        print('valid batches shape = {}'.format(self.valid_batches.shape))
        print('test batches shape= {}'.format(self.test_batches.shape))
        print([self.decoding_dict[x] for x in self.train_batches[0,:100]])
        
    def get_train_size(self):
            return self.num_train_batches

    def get_test_size(self):
        return self.num_test_batches

    def get_valid_size(self):
        return self.num_valid_batches

    def get_num_unique_chars(self):
        return len(set(self.text))


    def get_batches(self,batches):

        # iterate through the array, one sequence at a time
        for n in range(0, batches.shape[1], self.sequence_length):
            # The features
            #print('n = {}'.format(n))
            x = batches[:, n:n+self.sequence_length]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], batches[:, n+self.sequence_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], batches[:, 0]
            yield torch.tensor(x),torch.tensor(y)

        


class WordLanguageModelData(TextLanguageModelData):
    
    def __init__(self,textfiles,
                 batch_size=50,
                 sequence_length=80,
                 split_frac=0.1,
                 is_valid=True,
                 test_size=0.1,
                 encoding='utf-8',
                 preload=False,
                 cpu_factor=2,
                 tokens_file='tokens.pkl',
                 encoded_file='encoded_text.pkl',
                 encoding_dict_file='encoding_dict.pkl',
                 decoding_dict_file='decoding_dict.pkl'):
        
        raw_text = ''
        for file in textfiles:
            with codecs.open(file, 'r',encoding) as f:
                raw_text += f.read()
        print(len(raw_text))
        self.sequence_length = sequence_length
        self.raw_text = raw_text
        self.num_cores = int(get_num_cores()//cpu_factor)
        print('number of cores = {}'.format(self.num_cores))
       
        if preload:
            with open(tokens_file,'rb') as f:
                self.tokens = pickle.load(f)
        else:
            self.list_of_lines,self.list_of_ascii_lines = prepare_text(raw_text,
                                                                    self.num_cores)
            print('length of ascii lines = {},type = {}'.format(len(self.list_of_ascii_lines),
                                                         type(self.list_of_ascii_lines)))
            self.list_of_tokenized_lines = tokenize_lines(self.list_of_ascii_lines,
                                                        self.num_cores)
            self.tokens = flatten_list(self.list_of_tokenized_lines)
            with open(tokens_file,'wb') as f:
                pickle.dump(self.tokens,f)
        
        print('length of tokens = {}'.format(self.tokens))
        self.vocab,self.vocab_size = build_vocab(self.tokens)
        print('vocab size = {}'.format(self.vocab_size))
        print(self.tokens[:100])
        if preload:
            with open(encoding_dict_file,'rb') as f:
                self.encoding_dict = pickle.load(f)
            print('encoding dict loaded')
            with open(decoding_dict_file,'rb') as f:
                self.decoding_dict = pickle.load(f)
            print('decoding dict loaded')
        else:
            self._make_encoding_dict(encoding_dict_file)
            self._make_decoding_dict(decoding_dict_file)
        
        
        super().__init__(batch_size,is_valid=is_valid,test_size=test_size)

    def _make_encoding_dict(self,encoding_dict_file):
        self.encoding_dict = {word:number for number,word in enumerate(self.vocab,1)}
        with open(encoding_dict_file, 'wb') as f: 
            pickle.dump(self.encoding_dict,f)
        print('encoding dict created')
        
        
    def _make_decoding_dict(self,decoding_dict_file):
        self.decoding_dict = {number:word for word,number in self.encoding_dict.items()}
        with open(decoding_dict_file, 'wb') as f: 
            pickle.dump(self.decoding_dict,f)
        print('decoding dict created')

    
