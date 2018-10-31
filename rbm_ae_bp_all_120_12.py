# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from data import *
from tensorflow.python import pywrap_tensorflow
from RBM import *

# RBM_AE 120-12 所有节点 BP

seqdim = 120
maxepoch = 10
bpepoch = 200

rbm_out_dim1 = 100
rbm_out_dim2 = 50
rbm_out_dim3 = 25
rbm_out_dim4 = 12





for datanum in range(0, 59):
	if datanum == 0 or datanum == 5 or datanum == 57 or datanum == 45:
		print("Data %d is not exist" % datanum)
	else:
		train_set, test_set, max_train, min_train = data_preprocess_all(1, datanum)

		rbm_save_path = "/home/ljl/RBM_AE/save/rbm_ae_pre_all_120_12"
		rbm_node_save_path = rbm_save_path + "/rbm_ae_pre_" + str(datanum) + "_120_12"
		rbm1_save_path = rbm_node_save_path + "/rbm_1_pre_" + str(datanum) + "_120_12.npz"
		rbm2_save_path = rbm_node_save_path + "/rbm_2_pre_" + str(datanum) + "_120_12.npz"
		rbm3_save_path = rbm_node_save_path + "/rbm_3_pre_" + str(datanum) + "_120_12.npz"
		rbm4_save_path = rbm_node_save_path + "/rbm_4_pre_" + str(datanum) + "_120_12.npz"

		ckpt_save_path = "/home/ljl/RBM_AE/save/rbm_ae_bp_all_120_12/rbm_ae_bp_" + str(datanum) + "_120_12"
		ckpt_node_save_path = ckpt_save_path + "/rbm_ae_bp_" + str(datanum) + "_120_12.ckpt"

		if not os.path.exists(rbm_node_save_path):  # 如果不存在，则创造一个
			os.makedirs(rbm_node_save_path)
		if not os.path.exists(ckpt_save_path):  # 如果不存在，则创造一个
			os.makedirs(ckpt_save_path)

		# print('Pretraining Layer 1 with RBM: %d-%d ' % (seqdim, rbm_out_dim1))
		# w1_120_100, b1_1_100, b1_1_120, rbm_out_120_100 = rbm_train(train_set, seqdim, rbm_out_dim1, maxepoch)
		# np.savez(rbm1_save_path, w1_120_100=w1_120_100, b1_1_100=b1_1_100, b1_1_120=b1_1_120)
		#
		# print('Pretraining Layer 2 with RBM: %d-%d ' % (rbm_out_dim1, rbm_out_dim2))
		# w2_100_50, b2_1_50, b2_1_100, rbm_out_100_50 = rbm_train(rbm_out_120_100, rbm_out_dim1, rbm_out_dim2, maxepoch)
		# np.savez(rbm2_save_path, w2_100_50=w2_100_50, b2_1_50=b2_1_50, b2_1_100=b2_1_100)
		#
		# print('Pretraining Layer 3 with RBM: %d-%d ' % (rbm_out_dim2, rbm_out_dim3))
		# w3_50_25, b3_1_25, b3_1_50, rbm_out_50_25 = rbm_train(rbm_out_100_50, rbm_out_dim2, rbm_out_dim3, maxepoch)
		# np.savez(rbm3_save_path, w3_50_25=w3_50_25, b3_1_25=b3_1_25, b3_1_50=b3_1_50)
		#
		# print('Pretraining Layer 4 with RBM: %d-%d ' % (rbm_out_dim3, rbm_out_dim4))
		# w4, b4_en, b4_de, rbm_out_25_12 = rbm_train(rbm_out_50_25, rbm_out_dim3, rbm_out_dim4, maxepoch)
		# np.savez(rbm4_save_path, w4=w4, b4_en=b4_en, b4_de=b4_de)
		#
		# print('Retraining deep autoencoder with BP. ')
		# print('The BP uses %d epochs.' % bpepoch)
		# print('Now data node is %d.' % datanum)
		#
		rbm_ae_bp(train_set, bpepoch, ckpt_node_save_path,
				  file1=rbm1_save_path, file2=rbm2_save_path, file3=rbm3_save_path, file4=rbm4_save_path)

		test_error, prd = rbm_ae_bp_test(test_set, ckpt_node_save_path)

		test_err_oC = test_error * (max_train - min_train)
		print('Node %d test error %6.6f℃, PRD %6.6f' % (datanum, test_err_oC, prd))

		# f2 = open('/home/ljl/RBM_AE/log.txt', 'a')
		# f2.write('\n%6.6f,%6.6f ' % (test_err_oC, prd))
		# f2.close()

