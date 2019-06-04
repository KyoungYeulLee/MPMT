import pickle as pk
from scipy.cluster.hierarchy import fcluster

flag = 5 #&
min_d = float(flag) / 10 

with open('seq_linkage.bin', 'rb') as link_file:
	Z = pk.load(link_file)

Y = fcluster(Z, t=min_d, criterion='distance')

with open('tid_list_bs.bin', 'rb') as tid_file:
	tid_list = pk.load(tid_file)

cl2tars = {}
for tid,cl in zip(tid_list, Y):
	if cl not in cl2tars:
		cl2tars[cl] = []
	cl2tars[cl].append(int(tid)) 

with open('seq_cluster_d%d.bin'%flag, 'wb') as cl_file: 
	pk.dump(cl2tars, cl_file)

with open('seq_cluster_stat_d%d.txt'%flag, 'w') as cl_stat: 
	cl_list = sorted(cl2tars.keys(), key = lambda x: len(cl2tars[x]), reverse=True)
	for cl in cl_list:
		cl_stat.write("%d\t%d\n"%(cl,len(cl2tars[cl])))

