# /usr/bin/env python
# -*- coding:utf-8 -*-

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from random import choice


def create_index():

    docs_path = 'datasets/answer.txt'

    ACTIONS = []
    with open(docs_path,'r') as fr:
        docs = fr.readlines()
    print("Index num:",len(docs))

    for index,doc in enumerate(docs):
        action = {
            "_index": "yahooqa",
            "_type": "doc",
            "_id": str(index),
            "_source": {
                "contents":doc
            }
        }
        ACTIONS.append(action)

    es = Elasticsearch('127.0.0.1:9200')
    bulk(es, ACTIONS, index="yahooqa", raise_on_error=True) #批量创建索引
    return es


def search(eslink):
    query_path = 'selected_doc.txt'

    with open(query_path,'r') as fr:
        querys = fr.readlines()

    docs=[]

    for query in querys:
        doc = {
            "query": {
                "match": {
                    "contents": {
                        "query": query,
                        "operator": "or"
                    }

                }
            }
        }
        res = eslink.search(index="yahooqa", doc_type="doc", body=doc)
        match_docs = res['hits']['hits']

        ##match_num = res['hits']['total']
        for index in range(5):
            try:
                record = match_docs.pop(0)
                sample = record.get('_source').get('contents')
                docs.append(sample.strip())
            except:
                docs.append(choice(querys).strip())

    if len(docs)!=5*len(querys):
        print('正负样本数不平衡！')

    with open('selected_docs.txt','w') as fw:   #包括一个正样本和四个负样本
        fw.write('\n'.join(docs))


if __name__ == '__main__':

    es = create_index()
    search(es)