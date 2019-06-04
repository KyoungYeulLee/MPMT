import pickle as pk
import numpy as np

with open('tar2interact.bin', 'rb') as inter_file:
	tar2interact = pk.load(inter_file)
with open('fold2mols.bin', 'rb') as fm_file:
	fold2mols = pk.load(fm_file)

ori_tar_size = len(tar2interact)
remove_targets = set([])
for fold in range(1,4): # select targets with at least 1 active and 1 inactive data
	mols = set(fold2mols[fold])
	for tid in tar2interact: 
		act_num = len(tar2interact[tid][0] & mols)
		inact_num = len(tar2interact[tid][1] & mols)
		if act_num == 0 or inact_num == 0:
			remove_targets.add(tid)
for tid in remove_targets:
	del tar2interact[tid]
tar_list = list(tar2interact.keys())
with open('tar_list.bin', 'wb') as tar_file:
	pk.dump(tar_list, tar_file)
with open('tar_list.txt', 'w') as tar_file:
	for tar in tar_list:
		tar_file.write(str(tar)+'\n')
	
del mols
del fold2mols

with open('mol_list.bin', 'rb') as mol_file: # The order is same with mol2idx
	mol_list = pk.load(mol_file)

tar_size = len(tar_list)
print("The number of origianl targets = %d, removed targets = %d, remain_target = %d"%(ori_tar_size,len(remove_targets),tar_size))

tv_array = [] # build target activity vector
for mol in mol_list:
	target_vector = np.zeros(tar_size, dtype=int)
	for idx in range(tar_size):
		tid = tar_list[idx]
		if mol in tar2interact[tid][0]:
			target_vector[idx] = 1
		elif mol in tar2interact[tid][1]:
			target_vector[idx] = -1
	tv_array.append(target_vector) 
tv_array = np.array(tv_array) 
np.save("tv.npy", tv_array) # the order is same with fp.npy


