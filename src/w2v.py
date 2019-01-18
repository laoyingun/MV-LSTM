# /usr/bin/env python
# -*- coding:utf-8 -*-

from gensim.models.word2vec import Word2Vec
from load_data import preprocess
import pickle
import numpy as np
def generate_model():

    answer_dir = '../data/datasets/answer.txt'
    query_dir = '../data/datasets/query.txt'

    with open(answer_dir,'r') as fr1,open(query_dir,'r') as fr2:
        sentences = fr1.readlines()
        querys = fr2.readlines()
    sentences.extend(querys)
    sentences = [sen.split() for sen in sentences]

    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            sentences[i][j] = preprocess(sentences[i][j])

    model = Word2Vec(sentences, hs=1,min_count=1,window=5,size=100)
    model.save('../data/w2v/w2v.txt',)

def generate_initial_embedding():
    model_dir = '../data/w2v/w2v.txt'
    vocab_dir = '../data/vocab.txt'
    model = Word2Vec.load(model_dir).wv
    with open(vocab_dir,'r') as fr:
        words = fr.readlines()

    words = [word.strip()for word in words]
    initial = []

    for word in words:
        if word not in model.vocab:
            vector = np.random.uniform(-0.1,0.1,size=100)
            print('不应该呀！')
        else:
            vector = model[word]

        initial.append(vector)

    with open('../data/w2v/embedding','wb') as fw:
        pickle.dump(initial,fw)

def read_embedding():
    with open('../data/w2v/embedding','rb') as fr:
        embedding = pickle.load(fr)
    return embedding



if __name__ == '__main__':
    #generate_model()
    generate_initial_embedding()