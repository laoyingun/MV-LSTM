import tensorflow as tf
import os
from mv_lstm_model import MVLSTMConfig
from load_data import read_vocab,process_file
from main import evaluate


def test():

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess=sess,save_path=save_path)

    word2id = read_vocab(vocab_dir)
    query_test,doc_test = process_file(test_query_dir,test_docs_dir,word2id)

    print('Testing...')
    loss_test, p_1_test, mrr_test = evaluate(sess, query_test, doc_test)
    msg = 'Test Loss: {0:>6.2}, Test P@1: {1:>7.2%}, Test MRR: {2:8.2%}'
    print(msg.format(loss_test, p_1_test, mrr_test))


if __name__ == '__main__':

    vocab_dir = '../data/vocab.txt'

    test_query_dir = '../data/test/test_query.txt'
    test_docs_dir = '../data/test/test_docs.txt'

    config = MVLSTMConfig()
    # ['cosine','bilinear','tensor_layer']
    config.interaction = 'cosine'
    save_dir = './checkpoint/mvlstm'
    save_path = os.path.join(save_dir, config.interaction)  # 最佳验证结果保存路径
    test()

