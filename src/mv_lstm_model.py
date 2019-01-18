# /usr/env      python
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
from w2v import read_embedding
import numpy as np

class MVLSTMConfig(object):

    vocab_size = 30000
    embedding_dim = 100        #词向量维度
    hidden_dim = 50           #隐藏层维度
    batch_size = 64
    qa_learning_rate = 0.03
    sc_learning_rate = 0.3
    seq_length = 50
    num_epochs = 30
    keep_prob = 0.8           #mlp层神经元保留的比例
    interaction = 'tensor_layer'    #['cosine','bilinear','tensor_layer']
    unit_size = 5
    tensor_layer_num = 2      #tensor layer num
    save_per_batch = 50       #保存tensorboard
    topk = 5



class MVLSTM(object):

    def __init__(self,config):
        self.config  = config
        self.query = tf.placeholder(tf.int32,[None,self.config.seq_length])
        self.doc = tf.placeholder(tf.int32,[None,self.config.seq_length])
        self.mvlstm()

    def mvlstm(self):

        with tf.device('/cpu:0'):

            init = tf.constant_initializer(np.array(read_embedding()))
            embedding = tf.get_variable('embedding',[self.config.vocab_size,self.config.embedding_dim],initializer=init)
            query_embdding = tf.nn.embedding_lookup(embedding,self.query)
            doc_embedding  = tf.nn.embedding_lookup(embedding,self.doc)

        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            q_fw  = rnn.LSTMCell(num_units=self.config.hidden_dim)
            q_bw = rnn.LSTMCell(num_units=self.config.hidden_dim)
            (q_outputs,q_output_states) = tf.nn.bidirectional_dynamic_rnn(q_fw,q_bw,query_embdding,dtype=tf.float32)

        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            a_fw  = rnn.LSTMCell(num_units=self.config.hidden_dim)
            a_bw = rnn.LSTMCell(num_units=self.config.hidden_dim)
            (a_outputs, a_output_states) = tf.nn.bidirectional_dynamic_rnn(a_fw, a_bw, doc_embedding,dtype=tf.float32)
            
        """
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            fw = rnn.LSTMCell(num_units=self.config.hidden_dim)
            bw = rnn.LSTMCell(num_units=self.config.hidden_dim)
            (q_outputs, q_output_states) = tf.nn.bidirectional_dynamic_rnn(fw, bw, query_embdding, dtype=tf.float32)
            (a_outputs, a_output_states) = tf.nn.bidirectional_dynamic_rnn(fw, bw, doc_embedding, dtype=tf.float32)
        """

        with tf.name_scope('output'):
            q_output_fw , q_output_bw = q_outputs
            a_output_fw , a_output_bw = a_outputs

            q_output= tf.concat([q_output_fw,q_output_bw],axis=-1) #连接最后一个维度
            a_output= tf.concat([a_output_fw,a_output_bw],axis=-1)
            # q_fw_states,q_bw_states=q_output_states
            # a_fw_states,a_bw_states=a_output_states
            # q_hiddens = tf.concat([q_fw_states.h,q_bw_states.h],axis=-1)
            # a_hiddens = tf.concat([a_fw_states.h,a_bw_states.h],axis=-1)
        with tf.name_scope('interaction_tensor'):
            similarity = self.interaction(q_output,a_output,self.config.interaction)

        with tf.name_scope('k-max-pooling'):
            #to-dolist
            #需要把similarity矩阵reshape成[batch_size,-1]维度 的，然后计算出每个的前k个
            similarity = tf.reshape(similarity,[-1,self.simi_len(self.config.interaction)])
            mm_k = tf.nn.top_k(similarity,k=self.config.topk)[0]  #0是values，1是index

        with tf.name_scope('mlp'):
            mm_k = tf.layers.dropout(mm_k,self.config.keep_prob)
            logits = tf.layers.dense(mm_k,units=1,activation=tf.nn.sigmoid)
            self.y_pred = tf.reshape(logits,[-1,self.config.unit_size])

        with tf.name_scope('loss'):
            self.loss = self.ranking_loss(self.y_pred)
            self.optmz = tf.train.AdagradOptimizer(learning_rate=0.03).minimize(self.loss)

        with tf.name_scope('metrics'):
            self.p_1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_pred,1),0),tf.float32))

            _,ranks = tf.nn.top_k(self.y_pred,self.config.unit_size)
            # tf.where返回一个二维矩阵，第一列为true的行号，第二列会true的列号
            true_rank = tf.where(tf.equal(ranks,0))[:,1]
            self.mrr = tf.reduce_mean(1.0/(tf.cast(true_rank,tf.float32)+1.0))

    """
    Model interaction of
    positional sentence representation 
    """
    def interaction(self,u,v,arg):   #计算两个tensor的相似度
        if arg=='cosine':
            output = tf.einsum('abd,acd->abc',u,v)
            return output

        if arg=='bilinear':
            M = tf.Variable(tf.random_uniform((self.config.hidden_dim*2,self.config.hidden_dim*2),minval=-0.1,maxval=0.1))  #matrix to reweight uv
            tmp = tf.einsum('abd,dc->abc',u,M)       #uM
            output = tf.einsum('abd,acd->abc',tmp,v) #uMv
            return output
        #tf.matmul (a,b)  a,b的shape都一样必须为二维，如果三维第一维只能表示batch size
        if arg=='tensor_layer':
            Mt = tf.Variable(tf.random_uniform((self.config.tensor_layer_num,self.config.hidden_dim*2,self.config.hidden_dim*2),minval=-0.1,maxval=0.1))
            tmp = tf.einsum('abe,ced->abcd',u,Mt)
            output = tf.einsum('acbe,ade->abcd',tmp,v)
            return output

    def simi_len(self,arg='cosine'):  #需要把相似度矩阵拉伸成一维的然后求最大
        if arg=='cosine' or arg=='bilinear':
            return  self.config.seq_length*self.config.seq_length
        elif arg=='tensor_layer':
            return self.config.seq_length*self.config.seq_length*self.config.tensor_layer_num
        else:
            print('兄弟参数不对呀！')

    """
    ranking loss of 
    the model
    loss = max(0,1-s(SX,SY+)+s(SX,SY-))
    """

    def ranking_loss(self,y_pred):
        #for index in range(neg_sample_num):
        #    hinge_loss = 1-y_pred[::neg_sample_num+1]+y_pred[index+1::neg_sample_num+1]
        # loss = 1.0*(self.config.unit_size-1)+tf.reduce_sum(y_pred,1)-5*y_pred[:,0]
        # loss = tf.reduce_mean(tf.maximum(0.0,loss))
        #***************************************************************************
        # sum_loss = tf.Variable(0.0)
        # neg_num = self.config.unit_size-1
        # for index in range(neg_num):
        #     tmp = tf.maximum(0.0,1.0+y_pred[:,index+1]-y_pred[:,0])
        #     sum_loss += tf.reduce_mean(tmp)
        # return sum_loss/neg_num

        loss1 = tf.reduce_mean(tf.maximum(0.0, 1.0 + y_pred[:, 1] - y_pred[:, 0]))
        loss2 = tf.reduce_mean(tf.maximum(0.0, 1.0 + y_pred[:, 2] - y_pred[:, 0]))
        loss3 = tf.reduce_mean(tf.maximum(0.0, 1.0 + y_pred[:, 3] - y_pred[:, 0]))
        loss4 = tf.reduce_mean(tf.maximum(0.0, 1.0 + y_pred[:, 4] - y_pred[:, 0]))
        loss = tf.stack([loss1, loss2, loss3, loss4], axis=0)
        return tf.reduce_mean(loss)





