from fastcluster import linkage_vector
import pickle
import numpy as np

mol_list = []
ecfp_list = []
with open('ecfp4_data.txt') as ecfp_file:
	for el in ecfp_file:
		ew = el.strip('\n').split('\t')
		mol = int(ew[0])
		ecfp = ew[1]
		ecfp_bin = np.array([int(i) for i in ecfp], dtype=np.bool)
		mol_list.append(mol)
		ecfp_list.append(ecfp_bin)
ecfp_list = np.array(ecfp_list)

Z = linkage_vector(ecfp_list, method='single', metric='jaccard')

with open('mol_list.bin', 'wb') as mol_file:
	pickle.dump(mol_list, mol_file)

with open('linkage.bin', 'wb') as link_file:
	pickle.dump(Z, link_file)

