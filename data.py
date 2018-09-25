import os
import torch
import numpy as np
np.random.seed(5)
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word_count = {}
        self.pad_token = None
        self.sos_token = None
        self.eos_token = None
        self.unk_token = None

    def itos(self, word_idx):
        return self.idx2word[word_idx]

    def stoi(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[self.unk_token]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_count(self, word):
        if word not in self.word_count:
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1

    def count2dic(self, threshold=0):
        for item in self.word_count.items():
            if item[1] > threshold:
                self.idx2word.append(item[0])
                self.word2idx[item[0]] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, train_path, valid_path=None, test_path=None, min_len=5, max_len=20, lower=True):
        self.dictionary = Dictionary()
        self.dictionary.pad_token = '<pad>'
        self.dictionary.eos_token = '<eos>'
        self.dictionary.sos_token = '<sos>'
        self.dictionary.unk_token = '<unk>'
        self.min_len = min_len
        self.max_len = max_len
        self.lower = lower
        self.dictionary.add_word(self.dictionary.pad_token)
        self.dictionary.add_word(self.dictionary.sos_token)
        self.dictionary.add_word(self.dictionary.eos_token)
        self.dictionary.add_word(self.dictionary.unk_token)
        self.filter(train_path)
        if valid_path:
            self.filter(valid_path)
        if test_path:
            self.filter(test_path)
        self.dictionary.count2dic()
        self.train = self.tokenize(train_path)
        if valid_path:
            self.valid = self.tokenize(valid_path)
        if test_path:
            self.test = self.tokenize(test_path)

    def filter(self, path):
        """只保留句子长度在5～30以内的句子"""
        print(path)
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                if self.lower:
                    line = line.lower()
                line = line.strip()    
                sents = line.split("\t")
                lens = [len(x) for x in sents]
                if min(lens) > self.min_len and max(lens) < self.max_len:
                    for sent in sents:
                        sent = list(sent)
                        if len(sent) > self.min_len and len(sent) < self.max_len:
                            for word in sent:
                                self.dictionary.add_count(word)

    def tokenize(self, path):
        # Tokenize file content
        with open(path, 'r', encoding="utf-8") as f:
            ids = []
            for line in f:
                if self.lower:
                    line = line.lower()
                line = line.strip()
                sents = line.split("\t")
                lens = [len(x) for x in sents]
                if min(lens) > self.min_len and max(lens) < self.max_len:
                    d = [0] * (self.max_len + 2) * 5  # 包括首尾占位符
                    for j, sent in enumerate(sents[:-1]):
                        sent = list(sent)
                        words = ([] if self.dictionary.sos_token is None else [self.dictionary.sos_token]) + sent + \
                                ([] if self.dictionary.eos_token is None else [
                                 self.dictionary.eos_token])
                        for i, word in enumerate(words):
                            d[j*(self.max_len*2) + i] = self.dictionary.stoi(word)
                    last_sent = list(sents[-1])
                    words = ([] if self.dictionary.sos_token is None else [self.dictionary.sos_token]) + last_sent + \
                                ([] if self.dictionary.eos_token is None else [
                                 self.dictionary.eos_token])
                    for i, word in enumerate(words):
                        d[4*(self.max_len+2) + i] = self.dictionary.stoi(word)            
                    ids.append(d)
        return torch.tensor(ids, dtype=torch.long)

    def next_batch(self, bsz, mode="train"):
        if mode == "train":
            index = np.random.randint(0, self.train.size()[0], (bsz, ))
            chunk = self.train[index]
            context = chunk[:, :(self.max_len+2)*4]
            response = chunk[:, -(self.max_len+2):]
            return context, response
        elif mode == "valid":
            index = np.random.randint(0, self.valid.size()[0], (bsz, ))
            return self.valid[index]
        elif mode == "test":
            index = np.random.randint(0, self.test.size()[0], (bsz, ))
            chunk = self.test[index]
            context = chunk[:, :(self.max_len+2)*4]
            response = chunk[:, -(self.max_len+2):]
            return context, response
    def batch_num(self, bsz, mode="train"):
        if mode == "train":
            return len(self.train) // bsz
        elif mode == "valid":
            return len(self.valid) // bsz
        elif mode == "test":
            return len(self.test) // bsz


if __name__ == "__main__":
    corpus = Corpus("train.txt")
    for i in range(10):
        print(corpus.train[i])
