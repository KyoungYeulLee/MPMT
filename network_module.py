import tensorflow as tf
import numpy as np
import pickle as pk

def save_model(sess, save_path, save_meta=True, var_list=None): 
	if var_list is None:
		saver = tf.train.Saver()
	else:
		saver = tf.train.Saver(var_list)
	saver.save(sess, save_path, write_meta_graph=save_meta) 

def load_model(sess, save_path, var_list=None):
	if var_list is None:
		saver = tf.train.Saver()
	else:
		saver = tf.train.Saver(var_list)
	saver.restore(sess, save_path)

class EarlyStopping():
	def __init__(self, patience=0, reverse=False): 
		self.step = 0
		if reverse: 
			self.loss = -float('inf') 
		else: 
			self.loss = float('inf') 
		self.patience = patience
		self.reverse = reverse 
		self.stopped = False 
	
	def validate(self, loss):
		if self.reverse: 
			compared = self.loss > loss 
		else: 
			compared = self.loss < loss 
		if compared: 
			self.step += 1
			if self.step > self.patience:
				self.stopped = True 
				return 2 # termination code
			else:
				return 1 # continue code
		else:
			self.step = 0
			self.loss = loss
			return 0 # reset code

	def check_stop(self): 
		return self.stopped 

def batch_gen(input_array, output_array, idx_array, batch_size=100, tidx_array=None, known_only=False, use_all=False, weight=None): 
	if tidx_array is None: 
		tidx_array = np.array(range(output_array.shape[1])) 
	if known_only and len(tidx_array) != 1: 
		print("'Known only' option can be used only for a single target") 
		return None 
	elif known_only: 
		known_idx = np.where(abs(output_array[:,tidx_array])==1)[0] 
		new_idx = np.array(list(set(known_idx) & set(idx_array)), dtype=int) 
		if weight is not None: 
			new_idx = np.array(list(new_idx)*int(weight)) 
		fold = len(new_idx)/batch_size 
		if fold < 1: 
			multiple = int(np.ceil(1/fold)) 
			new_idx = np.array(list(new_idx)*multiple) 
			max_index = 1 
		else: 
			max_index = int(fold) 
	else:
		new_idx = np.array(list(idx_array), dtype=int) # make a novel idx array 
		if batch_size > len(new_idx): 
			batch_size = len(new_idx) 
		max_index = int(len(new_idx)/batch_size) 
	while True: 
		np.random.shuffle(new_idx)
		for i in range(max_index):
			is_last = (i == max_index-1)
			if is_last and use_all:
				idx = new_idx[batch_size*i:]
			else:
				idx = new_idx[batch_size*i:batch_size*(i+1)]
			input_batch = input_array[idx] 
			output_batch = output_array[idx][:,tidx_array] 
			yield input_batch, output_batch, is_last

