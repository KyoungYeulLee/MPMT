import pickle
import numpy as np

### compound feature vector (numpy)
fp_array = [] 
mol2idx = {} 
with open("semif_data.txt") as fp_file:
	idx = 0 
	for fl in fp_file:
		fw = fl.strip('\n').split('\t')
		mol = int(fw[0])
		fp = fw[1]
		fp_bin = np.array([int(i) for i in fp], dtype=np.bool)
		fp_array.append(fp_bin) 
		mol2idx[mol] = idx 
		if idx % 1000 == 0:
			print(idx)
		idx += 1 
fp_array = np.array(fp_array) 
np.save("fp.npy", fp_array) 
with open("mol2idx.bin", 'wb') as idx_file: 
	pickle.dump(mol2idx, idx_file) 
del fp_array
del mol2idx

### target taxonomy data (dictionary)
tax2tids = {}
with open("target_data.txt") as target_file:
	for fl in target_file:
		fw = fl.strip('\n').split('\t')
		tid = int(fw[0])
		tax = int(fw[1])
		if tax not in tax2tids:
			tax2tids[tax] = []
		tax2tids[tax].append(tid)
with open('tax2tids.bin', 'wb') as tax_file:
	pickle.dump(tax2tids, tax_file)

### interaction data (dictionary)
tar2interact = {}
with open("active_data.txt") as active_file, open("inactive_data.txt") as inactive_file:
	for fl in active_file:
		fw = fl.strip('\n').split('\t')
		tid = int(fw[0])
		mol = int(fw[1])
		if tid not in tar2interact:
			tar2interact[tid] = [set([]),set([])]
		tar2interact[tid][0].add(mol)
	for fl in inactive_file:
		fw = fl.strip('\n').split('\t')
		tid = int(fw[0])
		mol = int(fw[1])
		if tid not in tar2interact:
			print("target %d have no active data"%tid)
			tar2interact[tid] = [set([]),set([])]
		tar2interact[tid][1].add(mol)
remove_count = 0
remove_targets = [] 
for tid in tar2interact:
	confused_interact = tar2interact[tid][0] & tar2interact[tid][1] # remove interactions both active and inactive
	tar2interact[tid][0] = tar2interact[tid][0] - confused_interact
	tar2interact[tid][1] = tar2interact[tid][1] - confused_interact
	remove_count += 2*len(confused_interact)
	if len(tar2interact[tid][0]) == 0 or len(tar2interact[tid][1]) == 0: # targets having at least 1 active and 1 inactive data remains
		remove_targets.append(tid) 
for tid in remove_targets: 
	del tar2interact[tid] 
print("The number of removed interactions = %d"%remove_count)
print("The number of removed targets = %d"%len(remove_targets)) 
with open('tar2interact.bin', 'wb') as inter_file:
	pickle.dump(tar2interact, inter_file)

