# /usr/bin/env python
# -*- coding:utf-8 -*-
import random


def process():

    """
    dataset format:
    SX :
    SY+:
    SY-:
    """
    query_dir = 'datasets/query.txt'
    posdoc_dir = 'datasets/posdoc.txt'

    with open(query_dir,'r') as fr1,open(posdoc_dir,'r') as fr2:
        query = fr1.readlines()
        posdoc = fr2.readlines()

    print(len(query))
    print(len(posdoc))
    #筛选出q和p长度在5-50的问答对
    selected_query = []
    selected_posdoc= []


    for q,p in zip(query,posdoc):
        if len(q.split())>5 and len(q.split())<50 and len(p.split())>5 and len(p.split())<50:
            selected_query.append(q)
            selected_posdoc.append(p)
    print(len(selected_query))
    print(len(selected_posdoc))
    #打乱数据集顺序
    aggre = list(zip(selected_query,selected_posdoc))
    random.shuffle(aggre)
    query,posdoc = zip(*aggre)
    print('筛选后的的query个数:',len(query))
    print('筛选后的posdoc个数:',len(posdoc))

    return query,posdoc

#划分训练集、验证集、测试集
def divide():
    with open('selected_query.txt','r') as fr1,open('selected_docs.txt','r') as fr2:
        querys = fr1.readlines()
        docs = fr2.readlines()
    if len(querys)*5!=len(docs):
        print("QA不匹配！")


    query_len = len(querys)

    train_num = int(query_len*0.8)
    val_num = int(query_len*0.1)

    train_query = querys[:train_num]
    train_docs = docs[:train_num*5]
    val_query = querys[train_num:train_num+val_num]
    val_docs = docs[train_num*5:(train_num+val_num)*5]
    test_query=querys[train_num+val_num:]
    test_docs= docs[(train_num+val_num)*5:]

    if len(train_query)*5!=len(train_docs) or len(val_query)*5!=len(val_docs) or len(test_query)*5!=len(test_docs):
        print("Train , Validation ,Test 划分失败！")
    else:
        with open('train/train_query.txt','w') as fw1,open('train/train_docs.txt','w') as fw2,open('val/val_query.txt','w') as fw3,open('val/val_docs.txt','w') as fw4,open('test/test_query.txt','w') as fw5,open('test/test_docs.txt','w') as fw6:
            fw1.write(''.join(train_query))
            fw2.write(''.join(train_docs))
            fw3.write(''.join(val_query))
            fw4.write(''.join(val_docs))
            fw5.write(''.join(test_query))
            fw6.write(''.join(test_docs))



if __name__ == '__main__':

    # query,posdoc = process()
    # with open('selected_query.txt','w') as fr1,open('selected_doc.txt','w') as fr2:
    #     fr1.write(''.join(query))
    #     fr2.write(''.join(posdoc))
    divide()

















