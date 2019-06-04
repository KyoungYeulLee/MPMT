import pickle
from scipy.cluster.hierarchy import fcluster

min_d = 0.3 # distance cutoff for flat clustering

with open('linkage.bin', 'rb') as link_file:
	Z = pickle.load(link_file)

Y = fcluster(Z, t=min_d, criterion='distance')

with open('mol_list.bin', 'rb') as mol_file:
	mol_list = pickle.load(mol_file)

cl2mols = {}
for mol,cl in zip(mol_list, Y):
	if cl not in cl2mols:
		cl2mols[cl] = []
	cl2mols[cl].append(mol)

with open('cluster.bin', 'wb') as cl_file:
	pickle.dump(cl2mols, cl_file)

with open('cluster_stat.txt', 'w') as cl_stat:
	cl_list = sorted(cl2mols.keys(), key = lambda x: len(cl2mols[x]), reverse=True)
	for cl in cl_list:
		cl_stat.write("%d\t%d\n"%(cl,len(cl2mols[cl])))

