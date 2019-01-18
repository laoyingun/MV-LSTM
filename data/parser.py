# /usr/bin/env python
# -*- coding;utf-8 -*-

"""
处理xml问答数据

"""

import os
import xml.dom.minidom as xmldom
import re
def generate_data():
    query=list()
    posdoc=list()
    answer=list()
    file_name = 'datasets/manner.xml'
    data_path = os.path.abspath(file_name)
    #文档对象
    domobj = xmldom.parse(data_path)
    #得到元素对象
    elementobj = domobj.documentElement
    #获得子标签
    otheranswerobj = elementobj.getElementsByTagName('answer_item')
    for otheranswer in otheranswerobj:
        answer.append(re.sub(r'<br />|\n','',otheranswer.firstChild.data))

    subElementObj = elementobj.getElementsByTagName('vespaadd')
    for sub in subElementObj:
        subject = sub.getElementsByTagName('subject')[0].firstChild.data
        subject = re.sub(r'<br />|\n','',subject)
        best = sub.getElementsByTagName('bestanswer')[0].firstChild.data
        best = re.sub(r'<br />|\n', '', best)
        query.append(subject)
        posdoc.append(best)

    if len(query)!=len(posdoc):
        print('Parse Error!')
    print('query length:',len(query))
    print('positive length:',len(posdoc))
    with open('datasets/query.txt','w') as f1,open('datasets/posdoc.txt','w') as f2,open('datasets/answer.txt','w') as f4:

        f1.write('\n'.join(query))
        f2.write('\n'.join(posdoc))
        f4.write('\n'.join(answer))

if __name__ == '__main__':
    generate_data()





