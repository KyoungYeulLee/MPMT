import tensorflow as tf
from utils import uniform_initializer
from utils import identity_function
import pandas as pd
import numpy as np
import itertools
import os

def load_params():
	home_dir = os.environ.get('HOME')+'/deep_pred/'

	fixed = {
		'model_name': 'Single', 
		'multi_run': False, 
		'multi_mode': False, 
		'cluster_split': False, # True for MPMT
		'include_all': False, # Train targets not within clusters using single-task learning for MPMT.py
		'simultaneous': True, # Use train_and_test() function. True for singletask.py, MT_binary.py, MPMT.py
		'patience': 3, # For simultaneous=True, patience<=3 is recommended as it cannot save previous model
		'unknown_val': None, # value imposed for unknown interaction. 0~0.5 (0: negative, 1: positive), None for ignoring unknown
		'sign_bal': False, # balance between positive and negative data
		'batch_bal': False, # balance between classes in training batches (=task weighting)
		'use_all': True, # use all the data for 1 epoch training but batch size could be differentiated
		'performance_standard': 'loss', # 'recall', 'roc', 'pr', 'loss'. standard for early stopping and hyperparameter selection
		'validation_code': 1, # 0 for simple crossvalidation, 1 for nested crossvalidation
		'fast_inner': True, # use 1/10 test data for inner crossvalidation for time efficiency
		'mol_file': home_dir+'data/fold2mols.bin',
		'fp_file': home_dir+'data/fp.npy',
		'tv_file': home_dir+'data/tv.npy',
		'idx_file': home_dir+'data/mol2idx.bin',
		'validation_frac': 0.001, # 0.01 is recommended for seq_cluster_d2 with 'max' option as the validation set is too small 
		'target_code': 2, # 0 for all target use, 1 for a specific group of targets from cluster, 2 for a specific taxonomy, 3 for both cluster and taxonomy
		'tid_file': home_dir+'data/tar_list.bin', 
#		'cluster_file': home_dir+'data/seq_cluster_d2.bin', 
#		'cluster_select': 'max', # max: the cluster with maximum number of targets, diverse: One target from each cluster, all: all targets in clusters
		'tax_file': home_dir+'data/tax2tids.bin', 
		'tax_select': 'human', 
		'use_ext_file': False,
#		'ext_fp_file': None,
#		'ext_tv_file': None,
#		'ext_idx_file': None,
		'target_test': True, # calculate target-AUC for ROC_AUC
		'do_save': False, # for simultaneous=True, this option can raise time significantly.
		'save_dir': home_dir+'models/', # save directory for models
		'log_dir': home_dir+'log/', # if this variable equals None, just printing but not logging results
		'top_k': 11,
 		'max_epoch': 50,
		'min_epoch': 5,
		'eps': 1e-6, 
	}

	hyperparams = {
		'batch_size': [100],
		'learn_rate': [0.01, 0.001],
		'idropout': [0.2, 0.0],
		'dropout': [0.5, 0.0],
		'layers': [[2048,2048,2048], [1024,1024,1024], [4096,4096,4096]], 
		'hidden_initializer': [uniform_initializer],
		'hidden_activation': [tf.nn.relu],
		'output_initializer': [uniform_initializer],
		'output_activation': [identity_function],
		'optimizer': [tf.train.AdamOptimizer], 
	}

	hyperparams = pd.DataFrame(list(itertools.product(*hyperparams.values())), columns=hyperparams.keys())
	hyperparams.index = np.arange(len(hyperparams.index.values))
	return fixed, hyperparams
	
