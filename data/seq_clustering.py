from fastcluster import linkage 
import pickle as pk 
import numpy as np

dist_vec = np.load('dist_seq.npy')

Z = linkage(dist_vec, 'average') # UPGMA

with open('seq_linkage.bin', 'wb') as link_file: 
	pk.dump(Z, link_file)

