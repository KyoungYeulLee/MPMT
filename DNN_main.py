import tensorflow as tf
import argparse
import numpy as np
import pickle as pk
from random import randint
from utils import print_and_log
import os
import sys
sys.path.insert(0, 'hypers/')
sys.path.insert(0, 'architectures/') 
from MPMT import Network # architecture
import hyper_MPMT as hyper # hyperparameters

def main_run(fixed, hyperparams, pre_param, log_file):
	os.system("mkdir "+fixed['save_dir']+fixed['model_name']+'/')
	if fixed['multi_run']: 
		path = fixed['save_dir']+fixed['model_name']+'/%d/'%fixed['cluster_select'] 
		os.system("mkdir "+path) 
	## Target selection
	if fixed['target_code'] == 0: # use all targets
		with open(fixed['tid_file'], 'rb') as tid_file: 
			tid_list = pk.load(tid_file) 
		tidx_array = np.array([tidx for tidx,tid in enumerate(tid_list)]) 
	elif fixed['target_code'] == 1: # use a specific group of targets from cluster
		with open(fixed['tid_file'], 'rb') as tid_file: 
			tid_list = pk.load(tid_file)
		with open(fixed['cluster_file'], 'rb') as cls_file: 
			cl2tars = pk.load(cls_file) 
		if fixed['cluster_select'] == 'max': 
			cl_list = sorted(cl2tars.keys(), key = lambda x: len(cl2tars[x]), reverse=True) 
			selected_tars = cl2tars[cl_list[0]] 
		elif fixed['cluster_select'] == 'diverse': 
			selected_tars = [] 
			for cl in cl2tars: 
				selected_tars.append(cl2tars[cl][0]) 
		elif type(fixed['cluster_select']) == int: 
			selected_tars = cl2tars[fixed['cluster_select']] 
		tidx_array = np.array([tidx for tidx,tid in enumerate(tid_list) if tid in selected_tars]) 
	elif fixed['target_code'] == 2: # use specific taxonomy
		with open(fixed['tid_file'], 'rb') as tid_file: 
			tid_list = pk.load(tid_file) 
		with open(fixed['tax_file'], 'rb') as tax_file: 
			tax2tids = pk.load(tax_file) 
		if fixed['tax_select'] == 'human': 
			selected_tars = tax2tids[9606] 
		tidx_array = np.array([tidx for tidx,tid in enumerate(tid_list) if tid in selected_tars]) 
	elif fixed['target_code'] == 3:  # use both cluster and taxonomy
		with open(fixed['tid_file'], 'rb') as tid_file:
			tid_list = pk.load(tid_file)
		# cluster
		with open(fixed['cluster_file'], 'rb') as cls_file: 
			cl2tars = pk.load(cls_file) 
		if fixed['cluster_select'] == 'max': 
			cl_list = sorted(cl2tars.keys(), key = lambda x: len(cl2tars[x]), reverse=True) 
			cluster_tars = cl2tars[cl_list[0]] 
		elif fixed['cluster_select'] == 'diverse': 
			cluster_tars = [] 
			for cl in cl2tars: 
				cluster_tars.append(cl2tars[cl][0])
		elif fixed['cluster_select'] == 'all': 
			cluster_tars = []
			for cl in cl2tars:
				cluster_tars += cl2tars[cl] 
		elif type(fixed['cluster_select']) == int: 
			cluster_tars = cl2tars[fixed['cluster_select']] 
		# taxonomy
		with open(fixed['tax_file'], 'rb') as tax_file: 
			tax2tids = pk.load(tax_file) 
		if fixed['tax_select'] == 'human': 
			tax_tars = tax2tids[9606] 
		selected_tars = list(set(cluster_tars) & set(tax_tars)) 
		tidx_array = np.array([tidx for tidx,tid in enumerate(tid_list) if tid in selected_tars]) 
	
	if fixed['cluster_split']: 
		tidx_cluster = [] 
		tars_in_cluster = [] 
		with open(fixed['tid_file'], 'rb') as tid_file: 
			tid_list = pk.load(tid_file) 
		with open(fixed['cluster_file'], 'rb') as cls_file: 
			cl2tars = pk.load(cls_file) 
		for cl in cl2tars: 
			cls_tars = cl2tars[cl] 
			if len(cls_tars) < 2: 
				continue 
			cls_tidx = np.array([ntidx for ntidx,tidx in enumerate(tidx_array) if tid_list[tidx] in cls_tars], dtype=int) 
			tars_in_cluster += cls_tars 
			tidx_cluster.append(cls_tidx) 
		if fixed['include_all']: 
			single_clusters = [np.array([ntidx], dtype=int) for ntidx,tidx in enumerate(tidx_array) if tid_list[tidx] not in tars_in_cluster] 
			print_and_log("The number of clusters not within cluster file (single clusters) = %d"%len(single_clusters), log_file) 
			tidx_cluster += single_clusters 
		check_targets = np.array([]) 
		for cluster in tidx_cluster: 
			check_targets = np.concatenate((check_targets, cluster)) 
		print_and_log("The number of total targets in tidx_cluster = %d"%len(set(check_targets)), log_file) 
		fixed['tidx_cluster'] = tidx_cluster 

	if fixed['multi_mode']: 
		for i in range(len(tidx_cluster)): 
			path = fixed['save_dir']+fixed['model_name']+'/cls_%d/'%i 
			os.system("mkdir "+path) 
		print_and_log("Total number of cluster = %d"%len(tidx_cluster), log_file) 

	## Load data
	if fixed['validation_code'] <= 1: # cross-validation
		with open(fixed['mol_file'], 'rb') as mol_file:
			fold2mol = pk.load(mol_file)
		data, mol2idx, _, _ = construct_data(fixed, None, [], tidx_array, log_file)
		input_space = data[0].shape[1] 
		output_space = data[1].shape[1] 
		top_k = None 
		if fixed['top_k'] is None: 
			top_k = 0 
			for tv in data[1]: 
				num_tar = np.sum(np.maximum(tv,0)) 
				if top_k < num_tar: 
					top_k = num_tar 
			print_and_log("The top_k value is set to %d"%top_k, log_file) 
	elif fixed['validation_code'] == 2: # external validation
		train_data, mol2idx_train, test_data, mol2idx_test = construct_data(fixed, None, None, tidx_array, log_file) 

	## simple cross-validation
	if fixed['validation_code'] == 0:
		print_and_log("--- Simple cross-validation ---", log_file)
		overall_recall = 0.0 
		overall_roc = 0.0 
		overall_pr = 0.0 
		count = 0
		if pre_param is None:
			pre_param = randint(0, len(hyperparams.index)-1)
		print_and_log("The number for parameter selected is %d"%pre_param, log_file)
		params = hyperparams.iloc[pre_param]
		print_and_log(str(params), log_file)
		fold_keys = list(fold2mol.keys())
		for key in fold_keys:
			print_and_log("...Start fold %d..."%key, log_file)
			test_mols = fold2mol[key]
			train_mols = []
			for train_key in fold_keys:
				if train_key == key:
					continue
				train_mols += fold2mol[train_key]
			train_idx = build_idx_array(mol2idx, train_mols)
			test_idx = build_idx_array(mol2idx, test_mols)
			print_and_log("the number of train mol = %d"%len(train_idx), log_file) 
			print_and_log("the number of test mol = %d"%len(test_idx), log_file)
			weight = None 
			if fixed['batch_bal']: 
				weight = np.sum(np.absolute(data[1][train_idx]), axis=0) 	
			if fixed['multi_mode']: 
				roc_array = np.array([]) 
				pr_array = np.array([]) 
				for i,tidx_array in enumerate(tidx_cluster): 
					print_and_log("Start training for cluster num %d"%i, log_file) 
					new_model = Network(fixed, params, input_space, len(tidx_array), top_k, None, log_file, weight) 
					path = fixed['save_dir']+fixed['model_name']+'/cls_%d/auc_%d/'%(i, count) 
					os.system("mkdir "+path) 
					roc, pr = new_model.train_and_test(data, train_idx, test_idx, tidx_array, save_path=path) 
					roc_array = np.concatenate((roc_array, roc)) 
					pr_array = np.concatenate((pr_array, pr)) 
					reset_graph() 
				recall = 0.0 
				roc_auc = np.mean(roc_array) 
				pr_auc = np.mean(pr_array) 
				print_and_log("+++ test_set +++", log_file) 
				print_and_log("loss=%.5f, recall_top_%d=%.5f, ROC_AUC=%.5f, PR_AUC=%.5f"%(0.0, self.top_k, 0.0, roc_auc, pr_auc), log_file) 
				print_and_log("", log_file) 
				print_and_log("...Validation finished...", log_file) 
			else: 
				new_model = Network(fixed, params, input_space, output_space, top_k, None, log_file, weight) 
				if fixed['simultaneous']: 
					if fixed['multi_run']: 
						path = fixed['save_dir']+fixed['model_name']+'/%d/auc_%d/'%(fixed['cluster_select'], count) 
					else: 
						path = fixed['save_dir']+fixed['model_name']+'/auc_%d/'%count 
					if fixed['target_test']: 
						os.system("mkdir "+path) 
					else:
						path = None 
					_, recall, roc_auc, pr_auc = new_model.train_and_test(data, train_idx, test_idx, save_path=path) 
				else: 
					new_model.train(data, train_idx)
					if fixed['target_test']: 
						if fixed['multi_run']: 
							path = fixed['save_dir']+fixed['model_name']+'/%d/auc_%d/'%(fixed['cluster_select'], count) 
						else: 
							path = fixed['save_dir']+fixed['model_name']+'/auc_%d/'%count 
						os.system("mkdir "+fixed['save_dir']+fixed['model_name']+'/auc_%d/'%count) 
						_, recall, roc_auc, pr_auc = new_model.test_target(data, test_idx, save_path=fixed['save_dir']+fixed['model_name']+'/auc_%d/'%count) 
					else: 
						_, recall, roc_auc, pr_auc = new_model.test(data, test_idx) 
					reset_graph() 
			overall_recall += recall 
			overall_roc += roc_auc 
			overall_pr += pr_auc 
			count += 1
		overall_recall = overall_recall / count 
		overall_auc = overall_auc / count 
		overall_pr = overall_pr / count 
		print_and_log("Average Recall for this network = %.5f"%overall_recall, log_file) 
		print_and_log("Average ROC AUC for this network = %.5f"%overall_roc, log_file) 
		print_and_log("Average PR AUC for this network = %.5f"%overall_pr, log_file) 

	## Nested cross-validation
	if fixed['validation_code'] == 1:
		print_and_log("--- Nested cross-validation ---", log_file)
		outer_recall = 0.0 
		outer_roc = 0.0 
		outer_pr = 0.0 
		count_outer = 0
		fold_keys = list(fold2mol.keys())
		print_and_log("...Start outer cross-validation...", log_file)
		for key in fold_keys: # outer_cv
			print_and_log("...Start outer fold %d..."%key, log_file)
			validation_mols = fold2mol[key]
			learning_keys = []
			learning_mols = []
			for learn_key in fold_keys:
				if learn_key == key:
					continue
				learning_keys.append(learn_key)
				learning_mols += fold2mol[learn_key]
			param2score = {}
			for param_num in hyperparams.index:
				print_and_log("...Start inner cross-validation for parameter num %d..."%param_num, log_file)
				params = hyperparams.iloc[param_num]
				print_and_log(str(params), log_file)
				inner_value = 0.0 
				count_inner = 0
				for test_key in learning_keys: # inner_cv
					print_and_log("...Start inner fold %d..."%test_key, log_file)
					test_mols = fold2mol[test_key]
					train_mols = []
					for train_key in learning_keys:
						if train_key == test_key:
							continue
						train_mols += fold2mol[train_key]
					train_idx = build_idx_array(mol2idx, train_mols)
					test_idx = build_idx_array(mol2idx, test_mols)
					if fixed['fast_inner']: 
						test_idx = test_idx[0:int(len(test_idx)/10)] # To make learning time reasonable
					print_and_log("the number of train mol = %d"%len(train_idx), log_file) 
					print_and_log("the number of test mol = %d"%len(test_idx), log_file) 
					weight = None 
					if fixed['batch_bal']: 
						weight = np.sum(np.absolute(data[1][train_idx]), axis=0) 
					if fixed['multi_mode']: 
						roc_array = np.array([]) 
						pr_array = np.array([]) 
						for i,tidx_array in enumerate(tidx_cluster): 
							print_and_log("Start training for cluster num %d"%i, log_file) 
							new_model = Network(fixed, params, input_space, len(tidx_array), top_k, None, log_file, weight) 
							path = None 
							roc, pr = new_model.train_and_test(data, train_idx, test_idx, tidx_array, save_path=path) 
							roc_array = np.concatenate((roc_array, roc)) 
							pr_array = np.concatenate((pr_array, pr)) 
							reset_graph() 
						recall = 0.0 
						roc_auc = np.mean(roc_array) 
						pr_auc = np.mean(pr_array) 
						print_and_log("+++ test_set +++", log_file) 
						print_and_log("loss=%.5f, recall_top_%d=%.5f, ROC_AUC=%.5f, PR_AUC=%.5f"%(0.0, self.top_k, 0.0, roc_auc, pr_auc), log_file) 
						print_and_log("", log_file) 
						print_and_log("...Validation finished...", log_file) 
					else: 
						new_model = Network(fixed, params, input_space, output_space, top_k, None, log_file, weight) 
						if fixed['simultaneous']: 
							_, recall, roc_auc, pr_auc = new_model.train_and_test(data, train_idx, test_idx, save_path=None) 
						else: 
							new_model.train(data, train_idx) 
							_, recall, roc_auc, pr_auc = new_model.test(data, test_idx) 
						reset_graph()
					if fixed['performance_standard'] == 'recall': 
						value = recall 
					elif fixed['performance_standard'] == 'roc': 
						value = roc_auc 
					elif fixed['performance_standard'] == 'pr': 
						value = pr_auc 
					else:
						value = roc_auc 
					inner_value += value 
					count_inner += 1
				inner_value = inner_value / count_inner 
				param2score[param_num] = inner_value 
			best_param = max(param2score, key=param2score.get)
			print_and_log("The best parameter for the fold %d is parameter num %d"%(key, best_param), log_file)
			params = hyperparams.iloc[best_param]
			print_and_log(str(params), log_file)
			learn_idx = build_idx_array(mol2idx, learning_mols)
			validate_idx = build_idx_array(mol2idx, validation_mols)
			print_and_log("the number of learning mol = %d"%len(learn_idx), log_file) 
			print_and_log("the number of validation mol = %d"%len(validate_idx), log_file) 
			weight = None 
			if fixed['batch_bal']: 
				weight = np.sum(np.absolute(data[1][learn_idx]), axis=0) 
			if fixed['multi_mode']: 
				roc_array = np.array([]) 
				pr_array = np.array([]) 
				for i,tidx_array in enumerate(tidx_cluster): 
					print_and_log("Start training for cluster num %d"%i, log_file) 
					best_model = Network(fixed, params, input_space, len(tidx_array), top_k, None, log_file, weight) 
					path = fixed['save_dir']+fixed['model_name']+'/cls_%d/auc_%d/'%(i, count_outer) 
					os.system("mkdir "+path) 
					roc, pr = best_model.train_and_test(data, learn_idx, validate_idx, tidx_array, save_path=path) 
					roc_array = np.concatenate((roc_array, roc)) 
					pr_array = np.concatenate((pr_array, pr)) 
					reset_graph() 
				recall = 0.0 
				roc_auc = np.mean(roc_array) 
				pr_auc = np.mean(pr_array) 
				print_and_log("+++ test_set +++", log_file) 
				print_and_log("loss=%.5f, recall_top_%d=%.5f, ROC_AUC=%.5f, PR_AUC=%.5f"%(0.0, self.top_k, 0.0, roc_auc, pr_auc), log_file) 
				print_and_log("", log_file) 
				print_and_log("...Validation finished...", log_file) 
			else: 
				best_model = Network(fixed, params, input_space, output_space, top_k, None, log_file, weight) 
				if fixed['simultaneous']: 
					if fixed['multi_run']: 
						path = fixed['save_dir']+fixed['model_name']+'/%d/auc_%d/'%(fixed['cluster_select'], count_outer) 
					else: 
						path = fixed['save_dir']+fixed['model_name']+'/auc_%d/'%count_outer 
					if fixed['target_test']: 
						os.system("mkdir "+path) 
					else: 
						path = None 
					_, recall, roc_auc, pr_auc = best_model.train_and_test(data, learn_idx, validate_idx, save_path=path) 
				else: 
					best_model.train(data, learn_idx)
					if fixed['target_test']: 
						if fixed['multi_run']: 
							path = fixed['save_dir']+fixed['model_name']+'/%d/auc_%d/'%(fixed['cluster_select'], count_outer) 
						else: 
							path = fixed['save_dir']+fixed['model_name']+'/auc_%d/'%count_outer 
						os.system("mkdir "+path) 
						_, recall, roc_auc, pr_auc = best_model.test_target(data, validate_idx, save_path=path) 
					else: 
						_, recall, roc_auc, pr_auc = best_model.test(data, validate_idx)
				reset_graph()
			outer_recall += recall 
			outer_roc += roc_auc 
			outer_pr += pr_auc 
			count_outer += 1
		outer_recall = outer_recall / count_outer 
		outer_roc = outer_roc / count_outer 
		outer_pr = outer_pr / count_outer 
		print_and_log("Average Recall for this network = %.5f"%outer_recall, log_file)
		print_and_log("Average ROC AUC for this network = %.5f"%outer_roc, log_file)
		print_and_log("Average PR AUC for this network = %.5f"%outer_pr, log_file)

def reset_graph():
	tf.reset_default_graph()

def construct_data(fixed, train_mols=None, test_mols=[], tidx_array=None, log_file=None): 
	fp_array = np.load(fixed['fp_file'])
	tv_array = np.load(fixed['tv_file'])
	with open(fixed['idx_file'], 'rb') as idx_file:	
		mol2idx = pk.load(idx_file)
	mol_array = np.array(sorted(mol2idx.keys(), key=lambda x: mol2idx[x])) 
	print_and_log('length of mol_array = %d'%len(mol_array), log_file) 
	print_and_log('length of fp data = %d'%len(fp_array), log_file) 
	print_and_log('length of tv data = %d'%len(tv_array), log_file) 
	if tidx_array is not None: 
		tv_array = tv_array[:,tidx_array] 
	valid_idx_array = np.array([i for i,tv in enumerate(tv_array) if not np.all(tv==0)]) 
	if train_mols is None:
		print_and_log("'None' for train mols: All the molecules are included", log_file)
		idx_array = valid_idx_array
	else:
		train_idx_array = np.array([mol2idx[mol] for mol in train_mols], dtype=int)
		idx_array = np.array(list(set(train_idx_array) & set(valid_idx_array)), dtype=int) 
	train_data = (fp_array[idx_array], tv_array[idx_array]) 
	train_mol_array = mol_array[idx_array] 
	mol2idx_train = {mol:i for i,mol in enumerate(train_mol_array)} 
	print_and_log('The size of train_set:', log_file)
	print_and_log(str(train_data[0].shape), log_file)
	print_and_log(str(train_data[1].shape), log_file)
	if fixed['use_ext_file']: # for external validation, test_data is retrieved from this part
		fp_array = np.load(fixed['ext_fp_file'])
		tv_array = np.load(fixed['ext_tv_file'])
		with open(fixed['ext_idx_file'], 'rb') as idx_file:	
			mol2idx = pk.load(idx_file)
		mol_array = np.array(sorted(mol2idx.keys(), key=lambda x: mol2idx[x])) 
		if tidx_array is not None: 
			tv_array = tv_array[:,tidx_array] 
		valid_idx_array = np.array([i for i,tv in enumerate(tv_array) if not np.all(tv==0)]) 
	if test_mols is None:
		print_and_log("'None' for test mols: All the molecules are included", log_file)
		idx_array = valid_idx_array 
	else:
		test_idx_array = np.array([mol2idx[mol] for mol in test_mols], dtype=int)
		idx_array = np.array(list(set(test_idx_array) & set(valid_idx_array)), dtype=int) 
	test_data = (fp_array[idx_array], tv_array[idx_array]) 
	test_mol_array = mol_array[idx_array] 
	mol2idx_test = {mol:i for i,mol in enumerate(test_mol_array)} 
	print_and_log('The size of test_set:', log_file)
	print_and_log(str(test_data[0].shape), log_file)
	print_and_log(str(test_data[1].shape), log_file)
	return train_data, mol2idx_train, test_data, mol2idx_test

def build_idx_array(mol2idx, mol_list=None):
	if mol_list is None:
		mol_list = list(mol2idx.keys())
	idx_list = []
	for mol in mol_list:
		if mol not in mol2idx: 
			continue 
		idx_list.append(mol2idx[mol])
	return np.array(idx_list)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--pre_param', help='The number for pre-defined parameter among the parameter grid', default=None)
	args = parser.parse_args()
	fixed, hyperparams = hyper.load_params()
	pre_param = None
	if args.pre_param is not None:
		pre_param = int(args.pre_param)
	log_file = None
	if fixed['log_dir'] is not None:
		log_file = open(fixed['log_dir']+fixed['model_name']+'.log', 'w', 1)
	with os.popen("date") as pop_file:
		print_and_log(pop_file.readline(), log_file)
	main_run(fixed, hyperparams, pre_param, log_file)
	print_and_log('...Process done...', log_file)
	with os.popen("date") as pop_file:
		print_and_log(pop_file.readline(), log_file)
	if log_file is not None:
		log_file.close()


