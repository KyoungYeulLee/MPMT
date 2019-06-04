import os
import argparse
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem

parser = argparse.ArgumentParser(description="")
parser.add_argument('-t', '--ft_type', help="the type of features, 'f' for semiF, '4' for ecfp4, '6' for ecfp6. ex) 'f4' -> semiF & ecfp4 ")
args = parser.parse_args()

with open(os.environ.get("HOME")+"/deep_pred/data/molecule_std_data.txt") as m:
	out_f = open(os.environ.get("HOME")+"/deep_pred/data/semif_data.txt", 'w')
	err_f = open(os.environ.get("HOME")+"/deep_pred/data/semif_error.txt", 'w')
	out_4 = open(os.environ.get("HOME")+"/deep_pred/data/ecfp4_data.txt", 'w')
	err_4 = open(os.environ.get("HOME")+"/deep_pred/data/ecfp4_error.txt", 'w')
	out_6 = open(os.environ.get("HOME")+"/deep_pred/data/ecfp6_data.txt", 'w')
	err_6 = open(os.environ.get("HOME")+"/deep_pred/data/ecfp6_error.txt", 'w')

	fp_len = 2048
	check_spot_f = np.zeros((fp_len,), dtype=int)
 	check_spot_4 = np.zeros((fp_len,), dtype=int)
	check_spot_6 = np.zeros((fp_len,), dtype=int)

	for ml in m:
		mw = ml.strip('\n').split('\t')
		mol = mw[0]
		smi = mw[1]
		rd_mol = AllChem.MolFromSmiles(smi)
		if 'f' in args.ft_type:
			try:
				semif = Chem.RDKFingerprint(rd_mol, maxPath=6, fpSize=fp_len).ToBitString()
				out_f.write(mol+'\t'+semif+'\n')
				for i,fp in enumerate(semif):
					if fp == '1':
						check_spot_f[i] = 1
			except:
				err_f.write(mol+'\n')
		if '4' in args.ft_type:
			try:
				ecfp4 = AllChem.GetMorganFingerprintAsBitVect(rd_mol, 2, nBits=fp_len).ToBitString()
				out_4.write(mol+'\t'+ecfp4+'\n')
				for i,fp in enumerate(ecfp4):
					if fp == '1':
						check_spot_4[i] = 1
			except:
				err_4.write(mol+'\n')
		if '6' in args.ft_type:
			try:
				ecfp6 = AllChem.GetMorganFingerprintAsBitVect(rd_mol, 3, nBits=fp_len).ToBitString()
				out_6.write(mol+'\t'+ecfp6+'\n')
				for i,fp in enumerate(ecfp6):
					if fp == '1':
						check_spot_6[i] = 1
			except:
				err_6.write(mol+'\n')


	if 'f' in args.ft_type:
		print('The length of semiF feature = %d'%len(semif))
		print('The number of spots with real value among the whole features = %d'%np.sum(check_spot_f))
	if '4' in args.ft_type:
		print('The length of ecfp4 feature = %d'%len(ecfp4))
		print('The number of spots with real value among the whole features = %d'%np.sum(check_spot_4))
	if '6' in args.ft_type:
		print('The length of ecfp6 feature = %d'%len(ecfp6))
		print('The number of spots with real value among the whole features = %d'%np.sum(check_spot_6))

	out_f.close()
	err_f.close()
	out_4.close()
	err_4.close()
	out_6.close()
	err_6.close()


