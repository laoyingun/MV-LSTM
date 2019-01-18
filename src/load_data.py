# /usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr
import re


def build_vocab(dataset_dir, vocab_dir, vocab_size=30000):
    """根据训练集构建词汇表，存储"""
    answer_dir = '../data/datasets/answer.txt'
    query_dir = '../data/datasets/query.txt'

    with open(answer_dir, 'r') as fr1, open(query_dir, 'r') as fr2:
        sentences = fr1.readlines()
        querys = fr2.readlines()
    sentences.extend(querys)
    sentences = [sen.split() for sen in sentences]
    #列表生成式flatten，for循环位置和正常for循环位置一样
    all_data = [preprocess(word) for sen in sentences for word in sen if len(preprocess(word))>0]
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size)
    words, _ = list(zip(*count_pairs))
    words = list(words)
    with open(vocab_dir, mode='w') as fw:
        fw.write('\n'.join(words))

"""只保留字符串并且小写"""
def preprocess(data):

    return (''.join(re.split(r'[^A-Za-z]', data))).lower()


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir,'r') as fp:
        words = [word.strip()for word in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


# def to_words(content, words):
#     """将id表示的内容转换为文字"""
#     return ''.join(words[x] for x in content)
def process_file(query_dir,doc_dir ,word_to_id,max_length=50,unit_size=5):
    """将文件转换为id表示"""
    with open(query_dir,'r') as fr1,open(doc_dir,'r') as fr2:
        querys = fr1.readlines()
        docs = fr2.readlines()

    if len(querys)*unit_size!=len(docs):
        print('Process Error!!!!!!!')

    query_id, doc_id=[],[]

    for index,doc in enumerate(docs):
        query = querys[int(index/unit_size)]
        query_id.append([word_to_id[preprocess(word)] for word in query.split() if word in word_to_id])
        doc_id.append([word_to_id[preprocess(word)] for word in doc.split() if word in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    pad_query = kr.preprocessing.sequence.pad_sequences(query_id, max_length)
    pad_doc = kr.preprocessing.sequence.pad_sequences(doc_id,max_length)

    return pad_query,pad_doc


#batch_size表示每个batch中的query数
#实际数据数目为batch_size*num_units
def batch_iter(query, doc, batch_size=64,unit_size=5):
    """生成批次数据"""

    data_len = len(query)     #总数为query_num*5
    batch_data_size = batch_size*unit_size
    num_batch = int((data_len - 1) / batch_data_size) + 1
    # indices = np.random.permutation(np.arange(data_len))
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_data_size
        end_id = min((i + 1) * batch_data_size, data_len)
        yield query[start_id:end_id], doc[start_id:end_id]


if __name__ == '__main__':
    dataset_dir = '../data/datasets/answer.txt'
    vocab_dir = '../data/vocab.txt'
    build_vocab(dataset_dir,vocab_dir)