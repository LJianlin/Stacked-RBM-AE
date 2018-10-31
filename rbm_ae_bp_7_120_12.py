# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from data import *
from tensorflow.python import pywrap_tensorflow
from RBM import *


bpepoch = 200

train_set, test_set, max_train, min_train = data_preprocess(1)

print('Retraining deep autoencoder with BP. ')
print('The BP uses %d epochs.' % bpepoch)

ckpt_save_path = "/home/ljl/RBM_AE/save1/rbm_ae_bp_7_120_12/rbm_ae_bp_7_120_12.ckpt"

rbm_ae_bp(train_set, bpepoch, ckpt_save_path,
		  "/home/ljl/RBM_AE/save1/rbm_ae_pre_7_120_12/rbm_1_pre_120_12.npz",
		  "/home/ljl/RBM_AE/save1/rbm_ae_pre_7_120_12/rbm_2_pre_120_12.npz",
		  "/home/ljl/RBM_AE/save1/rbm_ae_pre_7_120_12/rbm_3_pre_120_12.npz",
		  "/home/ljl/RBM_AE/save1/rbm_ae_pre_7_120_12/rbm_4_pre_120_12.npz")

test_error = rbm_ae_bp_test(test_set, ckpt_save_path)

test_err_oC = test_error * (max_train - min_train)
print('Test error %6.6f℃' % test_err_oC)


