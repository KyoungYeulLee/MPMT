from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
import pickle as pk
import numpy as np

tid2seqs = {}
tid_list = []
with open('binding_sequences.txt') as bs_file:
	for fl in bs_file:
		fw = fl.strip('\n').split('\t')
		tid = fw[0]
		seq = fw[1]
		if tid not in tid2seqs:
			tid2seqs[tid] = []
			tid_list.append(tid)
		tid2seqs[tid].append(seq)
num_seq = len(tid_list)
num_pair = num_seq*(num_seq-1)/2
print(num_seq)
print(num_pair)
with open('tid_list_bs.bin', 'wb') as tid_file:
	pk.dump(tid_list, tid_file)

matrix = matlist.blosum62

tid2scores = {} 
for tid in tid_list:
	seqs = tid2seqs[tid]
	self_scores = []
	for seq in seqs:
		score = pairwise2.align.localdx(seq, seq, matrix, score_only=True)
		self_scores.append(score)
	tid2scores[tid] = self_scores
	
dist_list = []
for i1 in range(num_seq-1):
	tid1 = tid_list[i1]
	seqs1 = tid2seqs[tid1]
	scores1 = tid2scores[tid1] 
	for i2 in range(i1+1,num_seq):
		tid2 = tid_list[i2]
		seqs2 = tid2seqs[tid2]
		scores2 = tid2scores[tid2] 
		min_dist = float('inf')
		for m,seq1 in enumerate(seqs1): 
			score1 = scores1[m] 
			for n,seq2 in enumerate(seqs2): 
				score2 = scores2[n] 
				pair_score = pairwise2.align.localds(seq1, seq2, matrix, -1, -.1, score_only=True)
				new_dist = (score1 - pair_score) * (score2 - pair_score) / (score1 * score2)
				if new_dist < min_dist:
					min_dist = new_dist
		dist_list.append(min_dist)
print(len(dist_list))
dist_vec = np.array(dist_list)
np.save('dist_seq.npy', dist_vec)
