import tensorflow as tf
import numpy as np
import scipy
import math 
from scipy.stats import rankdata
from rdkit.Chem import AllChem as Chem
from sklearn import metrics

def print_and_log(new_line, log_file=None):
	print(new_line)
	if log_file is not None:
		log_file.write(new_line+'\n')

def set_aside_test(data, val_frac=0.1, no_overlap=False):
	if no_overlap:
		data = np.array(list(set(data))) 
	np.random.shuffle(data)
	boarder = int(len(data) * val_frac)
	if len(data.shape) == 1:
		return np.split(data, [boarder])
	elif len(data.shape) == 2:
		return np.vsplit(data, [boarder])
	# test, train

def get_recall(tars, tar_scores, top_k=11):
	sum_recall = 0.0
	sum_precision = 0.0
	sum_F1 = 0.0
	count = 0
	for i, score in enumerate(tar_scores):
		rank = len(score) - rankdata(score, method='ordinal') + 1
		ans = tars[i]
		lower_than_k = rank<=top_k
		true_target = ans>=1.0
		bigger_than_k = rank>top_k
		not_target = ans<=0.0
		tp = scipy.logical_and(lower_than_k, true_target)
		fn = scipy.logical_and(bigger_than_k, true_target)
		tn = scipy.logical_and(bigger_than_k, not_target)
		fp = scipy.logical_and(lower_than_k, not_target)
		num_tp = list(tp).count(True)
		num_fn = list(fn).count(True)
		num_tn = list(tn).count(True)
		num_fp = list(fp).count(True)
		if num_tp + num_fn == 0 or num_tp + num_fp == 0: 
			continue 
		recall = float(num_tp) / (num_tp + num_fn)
		precision = float(num_tp) / (num_tp + num_fp)
		if recall + precision <= 0:
			F1 = 0.0
		else:
			F1 = 2*precision*recall / (precision+recall)
		sum_recall += recall
		sum_precision += precision
		sum_F1 += F1
		count += 1
	ave_recall = sum_recall / count
	ave_precision = sum_precision / count
	ave_F1 = sum_F1 / count
	return ave_recall, ave_precision, ave_F1

def get_recall_list(tars, tar_scores, top_k=11): 
	recall_list = [] 
	for i, score in enumerate(tar_scores):
		rank = len(score) - rankdata(score, method='ordinal') + 1
		ans = tars[i]
		lower_than_k = rank<=top_k
		true_target = ans>=1.0
		bigger_than_k = rank>top_k
		not_target = ans<=0.0
		tp = scipy.logical_and(lower_than_k, true_target)
		fn = scipy.logical_and(bigger_than_k, true_target)
		tn = scipy.logical_and(bigger_than_k, not_target)
		fp = scipy.logical_and(lower_than_k, not_target)
		num_tp = list(tp).count(True)
		num_fn = list(fn).count(True)
		num_tn = list(tn).count(True)
		num_fp = list(fp).count(True)
		if num_tp + num_fn == 0 or num_tp + num_fp == 0: 
			continue 
		recall = float(num_tp) / (num_tp + num_fn)
		recall_list.append(recall) 
	return recall_list 

def get_curve_from_2d(tars, tar_scores, type_of_curve='roc', zero_is_ambiguous=False):
	tar_array = np.array([],dtype=int)
	score_array = np.array([],dtype=float)
	if zero_is_ambiguous:
		for i, tar in enumerate(tars):
			clear_point = np.where(tar!=0)[0]
			tar_array = np.concatenate((tar_array, tar[clear_point]))
			score_array = np.concatenate((score_array, tar_scores[i][clear_point]))
	else:
		for i,tar in enumerate(tars):
			tar_array = np.concatenate((tar_array, tar))
			score_array = np.concatenate((score_array, tar_scores[i]))
	if type_of_curve == 'roc':
		return metrics.roc_curve(tar_array, score_array)
		# fpr, tpr, thresholds
	if type_of_curve == 'pr':
		return metrics.precision_recall_curve(tar_array, score_array)
		# precision, recall, thresholds

def get_auc_from_2d(tars, tar_scores, zero_is_ambiguous=False):
	tar_array = np.array([],dtype=int)
	score_array = np.array([],dtype=float)
	if zero_is_ambiguous:
		for i, tar in enumerate(tars):
			clear_point = np.where(tar!=0)[0]
			tar_array = np.concatenate((tar_array, tar[clear_point]))
			score_array = np.concatenate((score_array, tar_scores[i][clear_point]))
	else:
		for i,tar in enumerate(tars):
			tar_array = np.concatenate((tar_array, tar))
			score_array = np.concatenate((score_array, tar_scores[i]))
	fpr, tpr, _ =  metrics.roc_curve(tar_array, score_array)
	precision, recall, _ = metrics.precision_recall_curve(tar_array, score_array)
	return metrics.auc(fpr, tpr), metrics.auc(recall, precision)
	# roc_auc, pr_auc

def get_array_from_2d(tars, tar_scores, zero_is_ambiguous=False): 
	tar_array = np.array([], dtype=int)
	score_array = np.array([], dtype=float)
	if zero_is_ambiguous:
		for i, tar in enumerate(tars):
			clear_point = np.where(tar!=0)[0]
			tar_array = np.concatenate((tar_array, tar[clear_point]))
			score_array = np.concatenate((score_array, tar_scores[i][clear_point]))
	else:
		for i,tar in enumerate(tars):
			tar_array = np.concatenate((tar_array, tar))
			score_array = np.concatenate((score_array, tar_scores[i]))
	return tar_array, score_array

def get_auc_from_array(tar_array, score_array): 
	row, col, _ =  metrics.roc_curve(tar_array, score_array)
	roc_auc = metrics.auc(row, col) 
	col, row, _ = metrics.precision_recall_curve(tar_array, score_array)
	pr_auc = metrics.auc(row, col)
	return roc_auc, pr_auc

def get_auc_per_col(tars, tar_scores, zero_is_ambiguous=False): 
	roc_auc_list = []
	pr_auc_list = []
	num_col = tars.shape[1] 
	for col in range(num_col): 
		tar_col = np.array(tars[:,col], dtype=int) 
		score_col = np.array(tar_scores[:,col], dtype=float) 
		if zero_is_ambiguous:
			clear_point = np.where(tar_col != 0) 
			tar_col = tar_col[clear_point]
			if len(tar_col) == 0: 
				print("There is no clear point for tidx %d"%col) 
				continue 
			elif len(tar_col[np.where(tar_col == 1)]) == 0: 
				print("There is no positive sample for tidx %d"%col) 
				continue 
			elif len(tar_col[np.where(tar_col == -1)]) == 0: 
				print("There is no negative sample for tidx %d"%col) 
				continue 
			score_col = score_col[clear_point]
		fpr, tpr, _ = metrics.roc_curve(tar_col, score_col)
		roc_auc = metrics.auc(fpr, tpr) 
		precision, recall, _ = metrics.precision_recall_curve(tar_col, score_col)
		pr_auc = metrics.auc(recall, precision) 
		roc_auc_list.append(roc_auc) 
		pr_auc_list.append(pr_auc) 
	return np.array(roc_auc_list), np.array(pr_auc_list)

def uniform_initializer(size_1, size_2):
	normalized_size = np.sqrt(6.0) / (np.sqrt(size_1 + size_2))
	return tf.random_uniform([size_1, size_2], minval=-normalized_size, maxval=normalized_size)

def gauss_initializer(size_1, size_2):
	return tf.random_normal([size_1, size_2], 0, 2. / (size_1 * size_2))

def identity_function(x):
	return x

