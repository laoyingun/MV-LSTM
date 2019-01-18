# ／usr/bin/env python
# -*- coding:utf-8 -*-

from mv_lstm_model import MVLSTM,MVLSTMConfig
from load_data import process_file,batch_iter,read_vocab
import tensorflow as tf
from tqdm import tqdm
import os


def feed_data(input_x,input_y):

    feed_dict={

        model.query:input_x,
        model.doc:input_y
    }
    return feed_dict


def evaluate(sess,query,doc):
    data_len = len(query)
    query_len = data_len/config.unit_size
    batch_val = batch_iter(query,doc,config.batch_size,config.unit_size)

    total_loss = 0.0
    total_p_1 = 0.0
    total_mrr = 0.0

    for val_query,val_doc in batch_val:
        loss,p_1,mrr = sess.run([model.loss,model.p_1,model.mrr],feed_dict=feed_data(val_query,val_doc))
        total_loss += loss*64
        total_p_1 += p_1*64
        total_mrr += mrr*64
    return total_loss/query_len,total_p_1/query_len,total_mrr/query_len


def train():

    word_2_id = read_vocab(vocab_dir)
    query_train,doc_train = process_file(train_query_dir,train_docs_dir,word_2_id)
    query_val,doc_val = process_file(val_query_dir,val_docs_dir,word_2_id)

    # 配置Tensorboard，重新训练时需要将文件夹删除，不然会覆盖
    tb_dir = './tensorboard/mvlstm-'+config.interaction
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    tf.summary.scalar('loss',model.loss)
    tf.summary.scalar('p@1',model.p_1)
    tf.summary.scalar('MRR',model.mrr)

    merged = tf.summary.merge_all()


    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sess = tf.Session()
    train_writer = tf.summary.FileWriter(tb_dir + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(tb_dir + '/test',sess.graph)
    sess.run(tf.global_variables_initializer())
    #writer.add_graph(sess.graph)

    best_val = 0.0
    total_batch = 0
    print('Training and Evaluation...')
    for epoch in tqdm(range(config.num_epochs)):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(query_train,doc_train,config.batch_size,config.unit_size)

        for query_batch,doc_batch in batch_train:

            feed_dict =feed_data(query_batch, doc_batch)
            val_feed = feed_data(query_val,doc_val)

            if total_batch%config.save_per_batch==0:
                s = sess.run(merged,feed_dict=feed_dict)
                train_writer.add_summary(s,total_batch)

            if total_batch%100==0:
                train_loss, train_p_1, train_mrr,y_pred= sess.run([model.loss, model.p_1, model.mrr,model.y_pred],feed_dict=feed_dict)
                val_loss, val_p_1, val_mrr = evaluate(sess, query_val, doc_val)
                s = sess.run(merged,feed_dict=val_feed)
                val_writer.add_summary(s,total_batch)
                #print(y_pred)
                #print(type(y_pred))
                #print(y_pred.shape)

                if val_p_1 >best_val:
                    best_val = val_p_1
                    saver.save(sess,save_path)
                    improved_str='*'
                else:
                    improved_str=''

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train p@1: {2:>7.2%}, Train MRR: {3:>7.2%} , Val Loss: {4:>6.2}, Val p@1: {5:>7.2%}, Val MRR: {6:>4.2%} {7}'
                print(msg.format(total_batch, train_loss, train_p_1, train_mrr,val_loss, val_p_1, val_mrr,improved_str))

            sess.run(model.optmz,feed_dict=feed_dict)
            total_batch+=1



if __name__ == '__main__':

    dataset_dir = '../data/answer.txt'
    vocab_dir = '../data/vocab.txt'
    train_query_dir = '../data/train/train_query.txt'
    train_docs_dir = '../data/train/train_docs.txt'
    val_query_dir = '../data/val/val_query.txt'
    val_docs_dir = '../data/val/val_docs.txt'


    config = MVLSTMConfig()
    save_dir = './checkpoint/' + config.interaction+'-mvlstm'
    save_path = os.path.join(save_dir, 'bestval')  # 最佳验证结果保存路径
    model = MVLSTM(config)
    train()
