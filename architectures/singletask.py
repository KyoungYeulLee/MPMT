import tensorflow as tf
import numpy as np
import os
from utils import print_and_log
from utils import set_aside_test
from utils import get_recall
from utils import get_recall_list 
from utils import get_auc_from_2d
from utils import get_auc_from_array 
from network_module import EarlyStopping
from network_module import batch_gen

class Network(object):
	def __init__(self, fixed, params, input_space, output_space, top_k=None, param_num=None, log_file=None, weight=None): 
		self.model_name = fixed['model_name']
		self.input_space = input_space 
		self.output_space = 1 
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
		self.batch_bal = fixed['batch_bal']
		self.standard = fixed['performance_standard'] 
		self.param_num = param_num
		self.log_file = log_file
		self.batch_size = params.batch_size
		self.check_train = False
		self.fp_tensor = tf.placeholder(tf.float32, [None, self.input_space])
		self.tar_tensor = tf.placeholder(tf.float32, [None, self.output_space]) 
		self.idropout = tf.placeholder_with_default(tf.constant(params.idropout, dtype=tf.float32), shape=())
		self.dropout = tf.placeholder_with_default(tf.constant(params.dropout, dtype=tf.float32), shape=())
		self.sign_weight = tf.placeholder(tf.float32, []) 
		self.do_save = fixed['do_save']
		
		## fully connected layers
		with tf.name_scope("target_predictor"):
			hidden_layers = params.layers
			sizes = [self.input_space] + hidden_layers + [self.output_space]
			self.last_size = sizes[-2] 
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
			if self.unknown_val is None: 
				flag = (self.tar_tensor+1.0)/2.0 
			else: 
				theta = self.unknown_val 
				flag = tf.nn.relu(self.tar_tensor*(1-theta)+theta) 
			rev_flag = tf.abs(flag-1.0)  
			self.init_w = tf.placeholder_with_default(params.output_initializer(sizes[-2], 2), shape=([sizes[-2], 2])) 
			self.init_b = tf.placeholder_with_default(tf.zeros([2]), shape=([2])) 
			self.w = tf.Variable(self.init_w, name='w') 
			self.b = tf.Variable(self.init_b, name='b') 
			cls = params.output_activation(tf.add(tf.matmul(predictor[-1], self.w), self.b)) 
			tar = tf.concat([flag,rev_flag], 1)
			self.cls_tensor = tf.split(tf.nn.softmax(cls, axis=1), [1,1], 1)[0] 
		softmax_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tar, logits=cls) 
		if self.sign_bal: 
			sign_mask = tf.nn.relu(self.tar_tensor) * (self.sign_weight-1) + 1 
			self.cls_loss = tf.reduce_sum(softmax_entropy*sign_mask) / tf.reduce_sum(sign_mask) 
		else: 
			self.cls_loss = tf.reduce_mean(softmax_entropy) 
		self.trainer = params.optimizer(params.learn_rate).minimize(self.cls_loss) 

	def train_and_test(self, train_data, idx_array=None, test_idx_array=None, tidx_array=None, save_path=None): # simultaneous learning and testing 
		if idx_array is None: 
			idx_array = np.array(range(len(train_data[0]))) 
		if test_idx_array is None: 
			test_idx_array = np.array(range(len(train_data[0]))) 
		if tidx_array is None: 
			tidx_array = np.array(range(train_data[1].shape[1])) 
		if self.sign_bal:
			if self.unknown_val is None: 
				num_pos = np.sum(np.maximum(train_data[1][idx_array],0), axis=0) 
				num_neg = np.sum(np.abs(train_data[1][idx_array]), axis=0) - num_pos 
			elif self.unknown_val == 0: 
				num_pos = np.sum(np.maximum(train_data[1][idx_array],0), axis=0)
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
			self.sign_ratio = np.ones(train_data[1].shape[1], dtype=np.float32) 
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		with tf.Session(config=config) as sess:
			## Initializing
			print_and_log("...Start initializing...", self.log_file)
			init = tf.global_variables_initializer()
			test_idx, train_idx = set_aside_test(idx_array, self.validation_frac)
			tidx2batches = {} 
			for tidx in tidx_array: 
				if self.unknown_val is None: 
					batches = batch_gen(train_data[0], train_data[1], train_idx, self.batch_size, np.array([tidx]), known_only=True, use_all=self.use_all) 
				else: 
					batches = batch_gen(train_data[0], train_data[1], train_idx, self.batch_size, np.array([tidx]), known_only=False, use_all=self.use_all) 
				tidx2batches[tidx] = batches 
			test_fp = train_data[0][test_idx]
			test_tv = train_data[1][test_idx]
			## Training and Testing
			print_and_log("...Start training...", self.log_file)
			rev = False 
			if save_path is None:
				tar_array = np.array([], dtype=np.int32) 
				score_array = np.array([], dtype=np.float32) 
			else:
				roc_array = [] 
				pr_array = [] 
			for i,tidx in enumerate(tidx_array): 
				batches = tidx2batches[tidx] 
				early_stopper = EarlyStopping(patience=self.patience, reverse=rev) 
				print_and_log("+++ for tidx=%d +++"%tidx, self.log_file) 
				for e in range(1, self.max_epoch+1): 
					if e == 1: 
						sess.run(init) 
					while True:
						batch_fp, batch_tv, is_last = next(batches)
						sess.run(self.trainer, feed_dict={self.fp_tensor: batch_fp, self.tar_tensor: batch_tv, self.sign_weight: self.sign_ratio[tidx]}) 
						if is_last:
							break
					# Testing
					if e >= self.min_epoch: 
						loss = sess.run(self.cls_loss, 
							feed_dict={self.fp_tensor: test_fp, self.tar_tensor: test_tv[:,np.array([tidx])], self.idropout: 0, self.dropout: 0, self.sign_weight: self.sign_ratio[tidx]}) 
						value = loss 
						early_stop_code = early_stopper.validate(value) 
						if early_stop_code == 0:
							pass
						if early_stop_code == 1:
							pass
						if early_stop_code == 2:
							print_and_log("...Terminating training by early stopper...", self.log_file)
							print_and_log("epoch=%d, loss=%.5f"%(e,loss), self.log_file) 
							print_and_log("", self.log_file)
							break
					if e == self.max_epoch:
						print_and_log("...Terminating training because it reaches to max epoch...", self.log_file)
						print_and_log("epoch=%d, loss=%.5f"%(e,loss), self.log_file) 
						print_and_log("", self.log_file)
				if self.do_save:
					saver.save(sess, self.save_dir+self.model_name+'/model_%d'%tidx)
				# Validating
				print_and_log("...Start testing...", self.log_file) 
				test_batch_size = 10000 
				batches_test = batch_gen(train_data[0], train_data[1], test_idx_array, test_batch_size, np.array([tidx]), False, True) 
				tars = np.array([], dtype=np.int32) 
				scores = np.array([], dtype=np.float32) 
				while True:
					batch_fp, batch_tar, is_last = next(batches_test) 
					batch_cls = sess.run(self.cls_tensor,
						feed_dict = {self.fp_tensor: batch_fp, self.tar_tensor: batch_tar, self.idropout: 0, self.idropout: 0}) 
					clear_point = np.where(batch_tar != 0) 
					tars = np.concatenate((tars, batch_tar[clear_point])) 
					scores = np.concatenate((scores, batch_cls[clear_point])) 
					if is_last: 
						break 
				if save_path is None: 
					tar_array = np.concatenate((tar_array, tars)) 
					score_array = np.concatenate((score_array, scores)) 
				else: 
					roc, pr = get_auc_from_array(tars, scores) 
					roc_array.append(roc) 
					pr_array.append(pr) 
			print_and_log("...Calculation finished...", self.log_file)
			if save_path is None: 
				roc_auc, pr_auc = get_auc_from_array(tar_array, score_array) 
			else: 
				roc_array = np.array(roc_array) 
				pr_array = np.array(pr_array) 
				np.save(save_path+'roc_auc.npy', roc_array) 
				np.save(save_path+'pr_auc.npy', pr_array) 
				roc_auc = np.mean(roc_array) 
				pr_auc = np.mean(pr_array) 
			# return
			print_and_log("+++ test_set +++", self.log_file)
			print_and_log("loss=%.5f, recall_top_%d=%.5f, ROC_AUC=%.5f, PR_AUC=%.5f"%(0, self.top_k, 0, roc_auc, pr_auc), self.log_file)
			print_and_log("", self.log_file)
			print_and_log("...Validation finished...", self.log_file)			
			with os.popen("date") as pop_file:
				print_and_log(pop_file.readline(), self.log_file)
			return 0, 0, roc_auc, pr_auc 

	def train(self, train_data, idx_array=None, tidx_array=None):
		return None

	def test(self, test_data, idx_array=None, tidx_array=None): 
		return None

	def test_target(self, test_data, idx_array=None, tidx_array=None, save_path=None): 
		return None


