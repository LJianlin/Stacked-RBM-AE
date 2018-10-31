# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow


def weight_prune(prune_file, prune_pro, prune_save_path):

	with tf.Session() as sess:
		reader = pywrap_tensorflow.NewCheckpointReader(prune_file)

		w1 = reader.get_tensor("w1_120_100")
		w2 = reader.get_tensor("w2_100_50")

		a = np.reshape(w1, [1, 12000])
		a = abs(a)
		a.sort()
		w1_120_100_thr = a[0, int(12000 * prune_pro)]
		for i in range(w1.shape[0]):
			for j in range(w1.shape[1]):
				if abs(w1[i, j]) < w1_120_100_thr:
					w1[i, j] = 0.

		b = np.reshape(w2, [1, 5000])
		b = abs(b)
		b.sort()
		w2_100_50_thr = b[0, int(5000 * prune_pro)]
		for i in range(w2.shape[0]):
			for j in range(w2.shape[1]):
				if abs(w2[i, j]) < w2_100_50_thr:
					w2[i, j] = 0.

		print('The prune proportion is ' + str(prune_pro * 100) + "%")

		w1_120_100 = tf.Variable(tf.convert_to_tensor(w1))
		b1_1_100 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b1_1_100")))
		b1_1_120 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b1_1_120")))
		w2_100_50 = tf.Variable(tf.convert_to_tensor(w2))
		b2_1_50 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b2_1_50")))
		b2_1_100 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b2_1_100")))
		w3_50_25 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w3_50_25")))
		b3_1_25 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b3_1_25")))
		b3_1_50 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b3_1_50")))
		w4 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w4")))
		b4_en = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b4_en")))
		b4_de = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b4_de")))

		sess.run(tf.global_variables_initializer())

		# add Saver ops
		saver = tf.train.Saver({'w1_120_100': w1_120_100, 'b1_1_100': b1_1_100, 'b1_1_120': b1_1_120,
								'w2_100_50': w2_100_50, 'b2_1_50': b2_1_50, 'b2_1_100': b2_1_100,
								'w3_50_25': w3_50_25, 'b3_1_25': b3_1_25, 'b3_1_50': b3_1_50,
								'w4': w4, 'b4_en': b4_en, 'b4_de': b4_de})

		saver.save(sess, prune_save_path)
		print('The weight after prune is saved.')


def retrain(train_set, steps, save_path, file):

	reader = pywrap_tensorflow.NewCheckpointReader(file)
	w1_120_100 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w1_120_100")))
	b1_1_100 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b1_1_100")))
	b1_1_120 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b1_1_120")))
	w2_100_50 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w2_100_50")))
	b2_1_50 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b2_1_50")))
	b2_1_100 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b2_1_100")))
	w3_50_25 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w3_50_25")))
	b3_1_25 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b3_1_25")))
	b3_1_50 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b3_1_50")))
	w4 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w4")))
	b4_en = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b4_en")))
	b4_de = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b4_de")))

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

	train_step = tf.train.AdamOptimizer(0.0002).minimize(regularized_loss)

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
			print("Retrain step %d | meanerror %6.6f sumerror %6.6f" % (step, errsum, errsum * batch_num))

		saver.save(sess, save_path)
		print('Retrained Model Saved.')
