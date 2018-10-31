# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from data import *
from tensorflow.python import pywrap_tensorflow
from RBM import *



seqdim = 120
maxepoch = 10

rbm_out_dim1 = 100
rbm_out_dim2 = 50
rbm_out_dim3 = 25
rbm_out_dim4 = 1

train_set, test_set, max_train, min_train = data_preprocess(1)

print('Pretraining a deep autoencoder. ')
print('The Science paper used 50 epochs. This uses %d ' % maxepoch)

# print('Pretraining Layer 1 with RBM: %d-%d ' % (seqdim, rbm_out_dim1))
# w1_120_100, b1_1_100, b1_1_120, rbm_out_120_100 = rbm_train(train_set, seqdim, rbm_out_dim1, maxepoch)
# np.savez("./save/rbm_ae_pre_7_120_1/rbm_1_pre_120_1.npz", w1_120_100=w1_120_100, b1_1_100=b1_1_100, b1_1_120=b1_1_120)
#
# print('Pretraining Layer 2 with RBM: %d-%d ' % (rbm_out_dim1, rbm_out_dim2))
# w2_100_50, b2_1_50, b2_1_100, rbm_out_100_50 = rbm_train(rbm_out_120_100, rbm_out_dim1, rbm_out_dim2, maxepoch)
# np.savez("./save/rbm_ae_pre_7_120_1/rbm_2_pre_120_1.npz", w2_100_50=w2_100_50, b2_1_50=b2_1_50, b2_1_100=b2_1_100)
#
# print('Pretraining Layer 3 with RBM: %d-%d ' % (rbm_out_dim2, rbm_out_dim3))
# w3_50_25, b3_1_25, b3_1_50, rbm_out_50_25 = rbm_train(rbm_out_100_50, rbm_out_dim2, rbm_out_dim3, maxepoch)
# np.savez("./save/rbm_ae_pre_7_120_1/rbm_3_pre_120_1.npz", w3_50_25=w3_50_25, b3_1_25=b3_1_25, b3_1_50=b3_1_50)
#
# print('Pretraining Layer 4 with RBM: %d-%d ' % (rbm_out_dim3, rbm_out_dim4))
# w4, b4_en, b4_de, rbm_out_25_12 = rbm_train(rbm_out_50_25, rbm_out_dim3, rbm_out_dim4, maxepoch)
# np.savez("./save/rbm_ae_pre_7_120_1/rbm_4_pre_120_1.npz", w4=w4, b4_en=b4_en, b4_de=b4_de)

test_error = rbm_ae_test(test_set, 1,
			"./save/rbm_ae_pre_7_120_1/rbm_1_pre_120_1.npz",
			"./save/rbm_ae_pre_7_120_1/rbm_2_pre_120_1.npz",
			"./save/rbm_ae_pre_7_120_1/rbm_3_pre_120_1.npz",
			"./save/rbm_ae_pre_7_120_1/rbm_4_pre_120_1.npz")

test_err_oC = test_error * (max_train - min_train)
print('Test error %6.6f℃' % test_err_oC)


