import time
import numpy as np
import tensorflow as tf

from models import GAT
from utils import process
from keyphrase_data.SemEval2010 import parse_input
import pickle

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'
train_file = 'data/train.pkl'
test_file = 'data/test.pkl'


dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 10
patience = 10
lr = 0.001  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [4] # numbers of hidden units per each attention head in each layer
n_heads = [4, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

#lgraphs, lgraph_features = None, None
# try:
#     file = open(train_file, 'rb')
#     pickle.load((lgraphs, lgraph_features), file)
# except IOError:
#     file = open(train_file, 'wb')
#     X = parse_input.preprocessor()
#     lgraphs, lgraph_features = X.extract('keyphrase_data/SemEval2010/train/', 'keyphrase_data/SemEval2010/train/train.combined.stem.final')
#     pickle.dump((lgraphs, lgraph_features), file)


X = parse_input.preprocessor()
lgraphs, lgraph_features = X.extract('keyphrase_data/SemEval2010/train/', 'keyphrase_data/SemEval2010/train/train.combined.final')

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = lgraph_features

print (adj.shape, features.shape, y_train.shape, y_val.shape, y_test.shape, train_mask.shape, val_mask.shape, test_mask.shape )

nb_nodes = features[0].shape[0]
ft_size = features[0].shape[1]
nb_classes = y_train[0].shape[1]

biases = adj

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    model=model()
    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    #log_resh = tf.Print(log_resh, [log_resh])
    #lab_resh = tf.Print(lab_resh, [lab_resh])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            step = 0
            tr_step = 0
            tr_size = features.shape[0]

            while step * batch_size < tr_size:
                if train_mask[step * batch_size:(step + 1) * batch_size][0][0] == 1.0:
                    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                        feed_dict={
                            ftr_in: features[step*batch_size:(step+1)*batch_size],
                            bias_in: biases[step*batch_size:(step+1)*batch_size],
                            lbl_in: y_train[step*batch_size:(step+1)*batch_size],
                            msk_in: train_mask[step*batch_size:(step+1)*batch_size],
                            is_train: True,
                            attn_drop: 0.6, ffd_drop: 0.6})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1
                step += 1

            step = 0
            vl_step = 0
            vl_size = features.shape[0]

            while step * batch_size < vl_size:
                if val_mask[step * batch_size:(step+1) * batch_size][0][0] == 1.0:
                    out_vl, loss_value_vl, acc_vl = sess.run([log_resh, loss, accuracy],
                        feed_dict={
                            ftr_in: features[step*batch_size:(step+1)*batch_size],
                            bias_in: biases[step*batch_size:(step+1)*batch_size],
                            lbl_in: y_val[step*batch_size:(step+1)*batch_size],
                            msk_in: val_mask[step*batch_size:(step+1)*batch_size],
                            is_train: False,
                            attn_drop: 0.0, ffd_drop: 0.0})
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1

                    unique, counts = np.unique(np.argmax(out_vl, axis=1), return_counts=True)
                    print (dict(zip(unique, counts)))

                step += 1



            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        step = 0
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            if test_mask[step * batch_size:(step + 1) * batch_size][0][0] == 1.0:
                out_ts, loss_value_ts, acc_ts = sess.run([log_resh, loss, accuracy],
                    feed_dict={
                        ftr_in: features[step*batch_size:(step+1)*batch_size],
                        bias_in: biases[step*batch_size:(step+1)*batch_size],
                        lbl_in: y_test[step*batch_size:(step+1)*batch_size],
                        msk_in: test_mask[step*batch_size:(step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1
            step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
