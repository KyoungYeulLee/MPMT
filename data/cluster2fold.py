import pickle

def tinest_fold(fold2mols):
	nf1 = len(fold2mols[1])
	nf2 = len(fold2mols[2])
	nf3 = len(fold2mols[3])
	if nf1 <= nf2 and nf1 <= nf3:
		return 1
	if nf2 <= nf1 and nf2 <= nf3:
		return 2
	if nf3 <= nf1 and nf3 <= nf2:
		return 3
	print "something is wrong"
	return 0

cl_items = []
with open('cluster_stat.txt') as cl_stat:
	for fl in cl_stat:
		fw = fl.strip('\n').split('\t')
		cl = int(fw[0])
		cl_len = int(fw[1])
		item = (cl,cl_len)
		cl_items.append(item)

with open('cluster.bin', 'rb') as cl_file:
	cl2mols = pickle.load(cl_file)

fold2mols = {1:[], 2:[], 3:[]}

for cl,cl_len in cl_items:
	des_fold = tinest_fold(fold2mols)
	fold2mols[des_fold] += cl2mols[cl]

print "Fold 1 : %d"%len(fold2mols[1])
print "Fold 2 : %d"%len(fold2mols[2])
print "Fold 3 : %d"%len(fold2mols[3])

with open('fold_mols.bin', 'wb') as fm_file:
	pickle.dump(fold2mols, fm_file)

