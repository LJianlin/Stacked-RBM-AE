# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from data import *
from tensorflow.python import pywrap_tensorflow
from RBM import *



seqdim = 120
maxepoch = 10
bpepoch = 200
retrainepoch = 10
prune_pro = 0.1

rbm_out_dim1 = 100
rbm_out_dim2 = 50
rbm_out_dim3 = 25
rbm_out_dim4 = 12

train_set, test_set, max_train, min_train = data_preprocess(1)

ckpt_save_path = "/home/ljl/RBM_AE/save/rbm_ae_bp_7_120_12/rbm_ae_bp_7_120_12.ckpt"
prune_save_path = "/home/ljl/RBM_AE/save/rbm_ae_prune_7_120_12/rbm_ae_prune_" + \
				  str(int(prune_pro * 100)) + "_7_120_12/rbm_ae_prune_" + str(int(prune_pro * 100)) + "_7_120_12.ckpt"
retrain_save_path = "/home/ljl/RBM_AE/save/rbm_ae_retrain_7_120_12/rbm_ae_retrain_" + \
					str(int(prune_pro * 100)) + "_7_120_12/rbm_ae_retrain_" + str(int(prune_pro * 100)) + "_7_120_12.ckpt"

if not os.path.exists(prune_save_path):  # 如果不存在，则创造一个
	os.makedirs(prune_save_path)
if not os.path.exists(retrain_save_path):  # 如果不存在，则创造一个
	os.makedirs(retrain_save_path)

# weight_prune(ckpt_save_path, prune_pro, prune_save_path)
# retrain(train_set, retrainepoch, retrain_save_path, prune_save_path)
# weight_prune(retrain_save_path, prune_pro, prune_save_path)
# retrain(train_set, retrainepoch, retrain_save_path, prune_save_path)
# weight_prune(retrain_save_path, prune_pro, prune_save_path)
# retrain(train_set, retrainepoch, retrain_save_path, prune_save_path)
# weight_prune(retrain_save_path, prune_pro, prune_save_path)


test_error = rbm_ae_bp_test(test_set, prune_save_path)

test_err_oC = test_error * (max_train - min_train)
print('Test error %6.6f℃' % test_err_oC)


