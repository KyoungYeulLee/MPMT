import tensorflow as tf
import numpy as np
import os
from utils import print_and_log
from utils import set_aside_test
from utils import get_recall
from utils import get_auc_from_2d
from utils import get_auc_per_col 
from network_module import EarlyStopping
from network_module import batch_gen

class Network(object):
	def __init__(self, fixed, params, input_space, output_space, top_k=None, param_num=None, log_file=None, weight=None): 
		self.model_name = fixed['model_name']
		self.input_space = input_space 
		self.output_space = output_space 
		self.validation_frac = fixed['validation_frac']
		if top_k is None: 
			self.top_k = fixed['top_k'] 
		else: 
			self.top_k = top_k 
		self.unknown_val = fixed['unknown_val'] 
		self.max_epoch = fixed['max_epoch']
		self.min_epoch = fixed['min_epoch']
		self.patience = fixed['patience']
		self.save_dir = fixed['save_dir']
		self.use_all = fixed['use_all'] 
		self.sign_bal = fixed['sign_bal'] 
		self.standard = fixed['performance_standard'] 
		self.weight = weight 
		if not fixed['batch_bal']: 
			self.weight = None 
		self.param_num = param_num
		self.log_file = log_file
		self.batch_size = params.batch_size
		self.check_train = False
		self.fp_tensor = tf.placeholder(tf.float32, [None, self.input_space])
		self.tar_tensor = tf.placeholder(tf.float32, [None, self.output_space])
		self.idropout = tf.placeholder_with_default(tf.constant(params.idropout, dtype=tf.float32), shape=())
		self.dropout = tf.placeholder_with_default(tf.constant(params.dropout, dtype=tf.float32), shape=())
		self.sign_weight = tf.placeholder(tf.float32, [self.output_space]) 
		self.compensate = tf.placeholder(tf.float32, [self.output_space]) 

		## fully connected layers
		with tf.name_scope("target_predictor"):
			hidden_layers = params.layers
			sizes = [self.input_space] + hidden_layers + [self.output_space]
			predictor = []
			# input layer
			self.fp_tensor = tf.nn.dropout(self.fp_tensor, 1.0-self.idropout)
			predictor.append(self.fp_tensor)
			# hidden layers
			for layer_num in range(len(hidden_layers)):
				w = tf.Variable(params.hidden_initializer(sizes[layer_num], sizes[layer_num+1]), name='w')
				b = tf.Variable(tf.zeros([sizes[layer_num+1]]), name='b')
				hidden_l = params.hidden_activation(tf.add(tf.matmul(predictor[-1], w), b), name='hidden_l')
				hidden_l = tf.nn.dropout(hidden_l, 1.0-self.dropout)
				predictor.append(hidden_l)
			# output layer
			w = tf.Variable(params.output_initializer(sizes[-2], sizes[-1]), name='w')
			b = tf.Variable(tf.zeros([sizes[-1]]), name='b')
			self.cls_tensor = params.output_activation(tf.add(tf.matmul(predictor[-1], w), b))
		if self.unknown_val is None: 
			sigmoid_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=(self.tar_tensor+1.0)/2.0, logits=self.cls_tensor) 
			mask = tf.abs(self.tar_tensor, name='mask')+fixed['eps'] 
			if self.sign_bal: 
				sign_mask = tf.nn.relu(self.tar_tensor) * (self.sign_weight - 1) + 1 
				self.cls_loss = tf.reduce_sum(sigmoid_entropy*mask*sign_mask) / tf.reduce_sum(mask*sign_mask) 
			else: 
				if self.weight is None: 
					self.cls_loss = tf.reduce_sum(sigmoid_entropy*mask) / tf.reduce_sum(mask) 
				else: 
					self.cls_loss = tf.reduce_sum(sigmoid_entropy*mask*self.compensate) / tf.reduce_sum(mask*self.compensate) 
		else: 
			theta = self.unknown_val 
			sigmoid_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.nn.relu(self.tar_tensor*(1-theta)+theta), logits=self.cls_tensor) 
			if self.sign_bal: 
				sign_mask = tf.nn.relu(self.tar_tensor) * (self.sign_weight - 1) + 1 
				self.cls_loss = tf.reduce_sum(sigmoid_entropy*sign_mask) / tf.reduce_sum(sign_mask) 
			else:
				self.cls_loss = tf.reduce_mean(sigmoid_entropy) 
		self.trainer = params.optimizer(params.learn_rate).minimize(self.cls_loss)

	def train(self, train_data, idx_array=None, tidx_array=None): 
		if idx_array is None: 
			idx_array = np.array(range(len(train_data[0]))) 
		if tidx_array is None: 
			tidx_array = np.array(range(train_data[1].shape[1]))
		if self.weight is not None: 
			compensate = 1 / self.weight 
		else:
			compensate = np.ones(train_data[1].shape[1]) 
		if self.sign_bal: 
			if self.unknown_val is None: 
				num_pos = np.sum(np.maximum(train_data[1][idx_array][:,tidx_array],0), axis=0) 
				num_neg = np.sum(np.abs(train_data[1][idx_array][:,tidx_array]), axis=0) - num_pos 
			elif self.unknown_val == 0: 
				num_pos = np.sum(np.maximum(train_data[1][idx_array][:,tidx_array],0), axis=0) 
				num_neg = len(idx_array) - num_pos 
			else:
				print_and_log("Error: Unknown data should be ignored or negative(0) for sign balancing", self.log_file) 
				return None 
			try: 
				self.sign_ratio = num_neg / num_pos 
			except:
				print_and_log("Error: Each target should have at least 1 positives and 1 negatives", self.log_file) 
				return None 
		else: 
			self.sign_ratio = np.ones(len(tidx_array), dtype=np.float32) 
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		saver = tf.train.Saver()
		with tf.Session(config=config) as sess:
			## Initializing
			print_and_log("...Start initializing...", self.log_file)
			init = tf.global_variables_initializer()
			sess.run(init) 
			test_idx, train_idx = set_aside_test(idx_array, self.validation_frac) 
			batches = batch_gen(train_data[0], train_data[1], train_idx, self.batch_size, tidx_array, use_all=self.use_all) 
			test_fp = train_data[0][test_idx]
			test_tv = train_data[1][test_idx][:,tidx_array] 
			## Training
			print_and_log("...Start training...", self.log_file)
			if self.standard in ['recall', 'roc', 'pr']: 
				rev = True 
			else: 
				rev = False 
			early_stopper = EarlyStopping(patience=self.patience, reverse=rev) 
			for e in range(1, self.max_epoch+1):
				while True:
					batch_fp, batch_tv, is_last = next(batches)
					sess.run(self.trainer, feed_dict={self.fp_tensor: batch_fp, self.tar_tensor: batch_tv, self.sign_weight: self.sign_ratio, self.compensate: compensate[tidx_array]}) 
					if is_last:
						break
				# Testing
				test_cls, loss = sess.run([self.cls_tensor, self.cls_loss], 
					feed_dict={self.fp_tensor: test_fp, self.tar_tensor: test_tv, self.idropout: 0, self.dropout: 0, self.sign_weight: self.sign_ratio, self.compensate: compensate[tidx_array]}) 
				recall, _, _ = get_recall(test_tv, test_cls, self.top_k) 
				roc_auc, pr_auc = get_auc_from_2d(test_tv, test_cls, zero_is_ambiguous=True) 
				print_and_log("+++ epoch=%i +++"%e, self.log_file)
				print_and_log("loss=%.5f, recall_top_%d=%.5f, ROC_AUC=%.5f, PR_AUC=%.5f"%(loss, self.top_k, recall, roc_auc, pr_auc), self.log_file)
				# Validation & Saving
				if e >= self.min_epoch:
					if self.standard == 'recall': 
						value = recall 
					elif self.standard == 'roc': 
						value = roc_auc 
					elif self.standard == 'pr': 
						value = pr_auc 
					else: 
						value = loss 
					early_stop_code = early_stopper.validate(value) 
					if early_stop_code == 0:
						print_and_log("...Saving current state...", self.log_file)
						if self.param_num is None:
							saver.save(sess, self.save_dir+self.model_name+'/model')
						else:
							saver.save(sess, self.save_dir+self.model_name+'/model_for_param_%d'%self.param_num)				
						self.check_train = True
					if early_stop_code == 1:
						pass
					if early_stop_code == 2:
						print_and_log("...Terminating training by early stopper...", self.log_file)
						print_and_log("", self.log_file)
						break
				if e == self.max_epoch:
					print_and_log("...Terminating training because it reaches to max epoch...", self.log_file)
				print_and_log("", self.log_file)
		with os.popen("date") as pop_file:
			print_and_log(pop_file.readline(), self.log_file)

	def test(self, test_data, idx_array=None, tidx_array=None): 
		if idx_array is None: 
			idx_array = np.array(range(len(test_data[0]))) 
		if tidx_array is None: 
			tidx_array = np.array(range(test_data[1].shape[1])) 
		if self.weight is not None: 
			compensate = 1 / self.weight 
		else:
			compensate = np.ones(test_data[1].shape[1]) 
		if not self.check_train:
			print_and_log("This model is not trained yet", self.log_file)
			return 0
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		saver = tf.train.Saver()
		with tf.Session(config=config) as sess:
			print_and_log("...Start testing...", self.log_file)
			if self.param_num is None:
				saver.restore(sess, self.save_dir+self.model_name+'/model')
			else:
				saver.restore(sess, self.save_dir+self.model_name+'/model_for_param_%d'%self.param_num)
			batches = batch_gen(test_data[0], test_data[1], idx_array, self.batch_size, tidx_array, use_all=True) 
			tar_con = np.array([[]]*self.output_space).T 
			cls_con = np.array([[]]*self.output_space).T 
			loss = 0.0
			count = 0
			while True:
				batch_fp, batch_tv, is_last = next(batches)
				batch_cls, batch_loss = sess.run([self.cls_tensor, self.cls_loss], 
					feed_dict={self.fp_tensor: batch_fp, self.tar_tensor: batch_tv, self.idropout: 0, self.dropout: 0, self.sign_weight: self.sign_ratio, self.compensate: compensate[tidx_array]}) 
				tar_con = np.concatenate((tar_con, batch_tv), axis=0)
				cls_con = np.concatenate((cls_con, batch_cls), axis=0)
				loss += batch_loss
				count += 1
				if is_last:
					break
			print_and_log("...Calculation finished...", self.log_file) 
			with os.popen("date") as pop_file: 
				print_and_log(pop_file.readline(), self.log_file) 
			loss /= count
			recall, _, _ = get_recall(tar_con, cls_con, self.top_k) 
			roc_auc, pr_auc = get_auc_from_2d(tar_con, cls_con, zero_is_ambiguous=True) 
			print_and_log("+++ test_set +++", self.log_file)
			print_and_log("loss=%.5f, recall_top_%d=%.5f, ROC_AUC=%.5f, PR_AUC=%.5f"%(loss, self.top_k, recall, roc_auc, pr_auc), self.log_file)
			print_and_log("", self.log_file)
			print_and_log("...Validation finished...", self.log_file) 
		with os.popen("date") as pop_file: 
			print_and_log(pop_file.readline(), self.log_file) 
		return loss, recall, roc_auc, pr_auc

	def test_target(self, test_data, idx_array=None, tidx_array=None, save_path=None): 
		if idx_array is None: 
			idx_array = np.array(range(len(test_data[0]))) 
		if tidx_array is None: 
			tidx_array = np.array(range(test_data[1].shape[1]))
		if not self.check_train:
			print_and_log("This model is not trained yet", self.log_file)
			return 0
		if self.weight is not None: 
			compensate = 1 / self.weight 
		else:
			compensate = np.ones(test_data[1].shape[1]) 
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		saver = tf.train.Saver()
		with tf.Session(config=config) as sess:
			print_and_log("...Start testing...", self.log_file)
			if self.param_num is None:
				saver.restore(sess, self.save_dir+self.model_name+'/model')
			else:
				saver.restore(sess, self.save_dir+self.model_name+'/model_for_param_%d'%self.param_num)
			batches = batch_gen(test_data[0], test_data[1], idx_array, self.batch_size, tidx_array, use_all=True) 
			tar_con = np.array([[]]*self.output_space).T 
			cls_con = np.array([[]]*self.output_space).T 
			loss = 0.0
			count = 0
			while True:
				batch_fp, batch_tv, is_last = next(batches)
				batch_cls, batch_loss = sess.run([self.cls_tensor, self.cls_loss], 
					feed_dict={self.fp_tensor: batch_fp, self.tar_tensor: batch_tv, self.idropout: 0, self.dropout: 0, self.sign_weight: self.sign_ratio, self.compensate: compensate[tidx_array]}) 
				tar_con = np.concatenate((tar_con, batch_tv), axis=0)
				cls_con = np.concatenate((cls_con, batch_cls), axis=0)
				loss += batch_loss
				count += 1
				if is_last:
					break
			print_and_log("...Calculation finished...", self.log_file) 
			with os.popen("date") as pop_file: 
				print_and_log(pop_file.readline(), self.log_file) 
			loss /= count
			recall, _, _ = get_recall(tar_con, cls_con, self.top_k)
			roc_array, pr_array = get_auc_per_col(tar_con, cls_con, zero_is_ambiguous=True) 
			if save_path is not None: 
				np.save(save_path+'roc_auc.npy', roc_array) 
				np.save(save_path+'pr_auc.npy', pr_array) 
			roc_auc = np.mean(roc_array) 
			pr_auc = np.mean(pr_array) 
			print_and_log("+++ test_set +++", self.log_file)
			print_and_log("loss=%.5f, recall_top_%d=%.5f, ROC_AUC=%.5f, PR_AUC=%.5f"%(loss, self.top_k, recall, roc_auc, pr_auc), self.log_file)
			print_and_log("", self.log_file)
			print_and_log("...Validation finished...", self.log_file) 
		with os.popen("date") as pop_file:
			print_and_log(pop_file.readline(), self.log_file)
		return loss, recall, roc_auc, pr_auc

	def train_and_test(self, train_data, idx_array=None, test_idx_array=None, tidx_array=None, save_path=None): 
		return None
