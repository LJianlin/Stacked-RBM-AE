# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np


def logistic(x):
    return 1.0 / (1 + np.exp(-x))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


def rbm_train(train_set, numdims, numhid, maxepoch):
    epsilonw = 0.1  # Learning rate for weights
    epsilonvb = 0.1  # Learning rate for biases of visible units
    epsilonhb = 0.1  # Learning rate for biases of hidden units
    weightcost = 0.0002  # 权重惩罚系数
    initialmomentum = 0.5  # 初始冲量
    finalmomentum = 0.9  # 最终冲量

    numcases = 1  # 每个batch样例数量=1
    numbatches = train_set.shape[0]  # batch的大小=362
    #numdims = train_set.shape[1]  # 可见层输入大小=120

    weights = 0.1 * np.random.random((numdims, numhid))  # [120, 100]
    hidbiases = np.zeros([1, numhid], np.float32)  # [1, 100]
    visbiases = np.zeros([1, numdims], np.float32)  # [1,120]
    pos_hidden_probs = np.zeros([numcases, numhid], np.float32)
    vishidinc = np.zeros([numdims, numhid], np.float32)
    hidbiasinc = np.zeros([1, numhid], np.float32)
    visbiasinc = np.zeros([1, numdims], np.float32)
    data = np.zeros([1, numdims], np.float32)  # [1,120]
    for epoch in range(maxepoch):

        errsum = 0

        for batch in range(numbatches):

            data[0, :] = train_set[batch, :]

            pos_hidden_activations = np.dot(data, weights) + hidbiases  # [1,120]*[120,100]+[1,100] = [1,100]
            pos_hidden_probs = logistic(pos_hidden_activations)  # sigmoid函数激活 得到隐藏层概率

            posprods = np.dot(data.T, pos_hidden_probs)  # [120,1]*[1,100]
            poshidact = sum(pos_hidden_probs)
            posvisact = sum(data)

            pos_hidden_states = pos_hidden_probs > np.random.rand(numcases, numhid)  # 吉布斯采样
            pos_hidden_states = pos_hidden_states.astype(np.float32)  # 转为float

            negdata = logistic(np.dot(pos_hidden_states, weights.T) + visbiases)  # [1,100]*[100,120]+[1,120] = [1,120]
            neg_hidden_probs = logistic(np.dot(negdata, weights) + hidbiases)
            negprods = np.dot(negdata.T, neg_hidden_probs)
            neghidact = sum(neg_hidden_probs)
            negvisact = sum(negdata)

            err_c = np.mean((abs(negdata - data)))
            errsum = err_c + errsum

            if epoch > 5:
                momentum = finalmomentum
            else:
                momentum = initialmomentum

            vishidinc = momentum * vishidinc + epsilonw * ((posprods - negprods)/numcases - weightcost * weights)
            visbiasinc = momentum * visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact)
            hidbiasinc = momentum * hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact)

            weights = weights + vishidinc
            visbiases = visbiases + visbiasinc
            hidbiases = hidbiases + hidbiasinc

        errsum = errsum / numbatches
        print('epoch %4i meanerror %6.6f sumerror %6.6f ' % (epoch, errsum, errsum * numbatches))

    return weights, hidbiases, visbiases, pos_hidden_probs


def rbm_ae_test(test_set, numhid, file1, file2, file3, file4):

    rbm1 = np.load(file1)
    rbm2 = np.load(file2)
    rbm3 = np.load(file3)
    rbm4 = np.load(file4)

    w1_120_100 = rbm1["w1_120_100"]
    b1_1_100 = rbm1["b1_1_100"]
    b1_1_120 = rbm1["b1_1_120"]
    w2_100_50 = rbm2["w2_100_50"]
    b2_1_50 = rbm2["b2_1_50"]
    b2_1_100 = rbm2["b2_1_100"]
    w3_50_25 = rbm3["w3_50_25"]
    b3_1_25 = rbm3["b3_1_25"]
    b3_1_50 = rbm3["b3_1_50"]
    w4 = rbm4["w4"]
    b4_en = rbm4["b4_en"]
    b4_de = rbm4["b4_de"]

    numbatches = test_set.shape[0]  # batch的大小=362
    numdims = test_set.shape[1]  # 可见层输入大小=120

    data = np.zeros([1, numdims], np.float32)  # [1,120]
    errsum = 0
    prd_c_sum = 0
    prd_d_sum = 0

    for batch in range(numbatches):

        data[0, :] = test_set[batch, :]

        pos_hidden_probs1 = logistic(np.dot(data, w1_120_100) + b1_1_100)
        pos_hidden_probs2 = logistic(np.dot(pos_hidden_probs1, w2_100_50) + b2_1_50)
        pos_hidden_probs3 = logistic(np.dot(pos_hidden_probs2, w3_50_25) + b3_1_25)
        pos_hidden_probs4 = logistic(np.dot(pos_hidden_probs3, w4) + b4_en)

        # pos_hidden_states = pos_hidden_probs4 > np.random.rand(1, numhid)  # 吉布斯采样
        # pos_hidden_states = pos_hidden_states.astype(np.float32)  # 转为float

        negdata4 = logistic(np.dot(pos_hidden_probs4, w4.T) + b4_de)  # [1,100]*[100,120]+[1,120] = [1,120]
        negdata3 = logistic(np.dot(negdata4, w3_50_25.T) + b3_1_50)
        negdata2 = logistic(np.dot(negdata3, w2_100_50.T) + b2_1_100)
        negdata = logistic(np.dot(negdata2, w1_120_100.T) + b1_1_120)

        err_c = np.mean(abs(negdata - data))
        errsum = err_c + errsum

        prd_c = np.square(err_c)
        prd_c_sum = prd_c + prd_c_sum
        prd_d = np.square(np.mean(abs(data)))
        prd_d_sum = prd_d + prd_d_sum

    err = errsum / numbatches
    prd = np.sqrt(prd_c_sum / prd_d_sum) * 100
    print("Test meanerror %6.6f sumerror %6.6f PRD %6.6f" % (err, errsum, prd))

    return err


def rbm_ae_bp(train_set, steps, save_path, file1, file2, file3, file4):

    rbm1 = np.load(file1)
    rbm2 = np.load(file2)
    rbm3 = np.load(file3)
    rbm4 = np.load(file4)

    w1_120_100 = tf.Variable(tf.to_float(tf.constant(rbm1["w1_120_100"])))
    b1_1_100 = tf.Variable(tf.to_float(tf.constant(rbm1["b1_1_100"])))
    b1_1_120 = tf.Variable(tf.to_float(tf.constant(rbm1["b1_1_120"])))
    w2_100_50 = tf.Variable(tf.to_float(tf.constant(rbm2["w2_100_50"])))
    b2_1_50 = tf.Variable(tf.to_float(tf.constant(rbm2["b2_1_50"])))
    b2_1_100 = tf.Variable(tf.to_float(tf.constant(rbm2["b2_1_100"])))
    w3_50_25 = tf.Variable(tf.to_float(tf.constant(rbm3["w3_50_25"])))
    b3_1_25 = tf.Variable(tf.to_float(tf.constant(rbm3["b3_1_25"])))
    b3_1_50 = tf.Variable(tf.to_float(tf.constant(rbm3["b3_1_50"])))
    w4 = tf.Variable(tf.to_float(tf.constant(rbm4["w4"])))
    b4_en = tf.Variable(tf.to_float(tf.constant(rbm4["b4_en"])))
    b4_de = tf.Variable(tf.to_float(tf.constant(rbm4["b4_de"])))

    batch_num = train_set.shape[0]  # batch的大小=362
    input_dim = train_set.shape[1]  # 可见层输入大小=120

    x = tf.placeholder("float", shape=[None, input_dim])
    l2_loss = tf.constant(0.0)
    data = np.zeros([1, input_dim], np.float32)
    lam = 0.0002

    l2_loss += tf.nn.l2_loss(w1_120_100)
    l2_loss += tf.nn.l2_loss(w2_100_50)
    l2_loss += tf.nn.l2_loss(w3_50_25)
    l2_loss += tf.nn.l2_loss(w4)

    hidden_encoder = tf.nn.sigmoid(tf.matmul(x, w1_120_100) + b1_1_100)
    hidden_encoder2 = tf.nn.sigmoid(tf.matmul(hidden_encoder, w2_100_50) + b2_1_50)
    hidden_encoder3 = tf.nn.sigmoid(tf.matmul(hidden_encoder2, w3_50_25) + b3_1_25)
    encoder_out = tf.nn.sigmoid(tf.matmul(hidden_encoder3, w4) + b4_en)

    hidden_decoder1 = tf.nn.sigmoid(tf.matmul(encoder_out, tf.transpose(w4)) + b4_de)
    hidden_decoder2 = tf.nn.sigmoid(tf.matmul(hidden_decoder1, tf.transpose(w3_50_25)) + b3_1_50)
    hidden_decoder3 = tf.nn.sigmoid(tf.matmul(hidden_decoder2, tf.transpose(w2_100_50)) + b2_1_100)
    out_decoder = tf.nn.sigmoid(tf.matmul(hidden_decoder3, tf.transpose(w1_120_100)) + b1_1_120)

    loss = tf.reduce_sum((out_decoder - x) * (out_decoder - x))
    regularized_loss = loss + lam * l2_loss
    xloss = tf.reduce_mean(abs(out_decoder - x))

    train_step = tf.train.AdamOptimizer(0.0001).minimize(regularized_loss)

    # add Saver ops
    saver = tf.train.Saver({'w1_120_100': w1_120_100,
                            'b1_1_100': b1_1_100, 'b1_1_120': b1_1_120,
                            'w2_100_50': w2_100_50, 'b2_1_50': b2_1_50, 'b2_1_100': b2_1_100,
                            'w3_50_25': w3_50_25, 'b3_1_25': b3_1_25, 'b3_1_50': b3_1_50,
                            'w4': w4, 'b4_en': b4_en, 'b4_de': b4_de
                            })
    errsum = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(0, steps):
            for batch in range(batch_num):

                data[0, :] = train_set[batch, :]
                _, cur_loss, xloss1 = sess.run([train_step, loss, xloss], feed_dict={x: data})
                errsum = errsum + xloss1

            errsum = errsum / batch_num
            print("Step %d | meanerror %6.6f sumerror %6.6f" % (step, errsum, errsum * batch_num))

        saver.save(sess, save_path)
        print('Trained Model Saved.')


def rbm_ae_bp_test(test_set, file):

    reader = pywrap_tensorflow.NewCheckpointReader(file)
    w1_120_100 = tf.convert_to_tensor(reader.get_tensor("w1_120_100"))
    b1_1_100 = tf.convert_to_tensor(reader.get_tensor("b1_1_100"))
    b1_1_120 = tf.convert_to_tensor(reader.get_tensor("b1_1_120"))
    w2_100_50 = tf.convert_to_tensor(reader.get_tensor("w2_100_50"))
    b2_1_50 = tf.convert_to_tensor(reader.get_tensor("b2_1_50"))
    b2_1_100 = tf.convert_to_tensor(reader.get_tensor("b2_1_100"))
    w3_50_25 = tf.convert_to_tensor(reader.get_tensor("w3_50_25"))
    b3_1_25 = tf.convert_to_tensor(reader.get_tensor("b3_1_25"))
    b3_1_50 = tf.convert_to_tensor(reader.get_tensor("b3_1_50"))
    w4 = tf.convert_to_tensor(reader.get_tensor("w4"))
    b4_en = tf.convert_to_tensor(reader.get_tensor("b4_en"))
    b4_de = tf.convert_to_tensor(reader.get_tensor("b4_de"))

    batch_num = test_set.shape[0]  # batch的大小=362
    input_dim = test_set.shape[1]  # 可见层输入大小=120

    x = tf.placeholder("float", shape=[None, input_dim])
    data = np.zeros([1, input_dim], np.float32)

    hidden_encoder = tf.nn.sigmoid(tf.matmul(x, w1_120_100) + b1_1_100)
    hidden_encoder2 = tf.nn.sigmoid(tf.matmul(hidden_encoder, w2_100_50) + b2_1_50)
    hidden_encoder3 = tf.nn.sigmoid(tf.matmul(hidden_encoder2, w3_50_25) + b3_1_25)
    encoder_out = tf.nn.sigmoid(tf.matmul(hidden_encoder3, w4) + b4_en)

    hidden_decoder1 = tf.nn.sigmoid(tf.matmul(encoder_out, tf.transpose(w4)) + b4_de)
    hidden_decoder2 = tf.nn.sigmoid(tf.matmul(hidden_decoder1, tf.transpose(w3_50_25)) + b3_1_50)
    hidden_decoder3 = tf.nn.sigmoid(tf.matmul(hidden_decoder2, tf.transpose(w2_100_50)) + b2_1_100)
    out_decoder = tf.nn.sigmoid(tf.matmul(hidden_decoder3, tf.transpose(w1_120_100)) + b1_1_120)

    loss = tf.reduce_mean(abs(out_decoder - x))

    prd_c = tf.square(loss)
    prd_d = tf.square(tf.reduce_mean(abs(x)))

    errsum = 0
    prd_c_sum = 0
    prd_d_sum = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for batch in range(batch_num):

            data[0, :] = test_set[batch, :]

            xloss, prdc, prdd = sess.run([loss, prd_c, prd_d], feed_dict={x: data})
            errsum = errsum + xloss

            prd_c_sum = prdc + prd_c_sum
            prd_d_sum = prdd + prd_d_sum


        err = errsum / batch_num
        prd = np.sqrt(prd_c_sum / prd_d_sum) * 100
        print("Test meanerror %6.6f sumerror %6.6f PRD %6.6f" % (err, errsum, prd))

    return err, prd