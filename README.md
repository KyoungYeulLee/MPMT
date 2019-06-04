# MPMT
data/ : data files and data processing python codes. The codes should be executed in advance in the order below.
   smiles2fp.py, mol_clustering.py, mol_flat_clusters.py, cluster2fold.py, data_generation.py, activity_vector.py, seq_normalized_distance.py, seq_clustering.py, seq_flat_clusters.py
   The file "molecule_std_data.smi.7z" should be unzipped.
architectures/ : deep neural architectures including network structure, train functions, and test functions.
hypers/ : hyperparameters including fixed parameters and hyperparameter grid for grid search.
DNN_main.py : main code for train and test deep neural networks. importing architecture and hyperparameters can be changed.
network_module.py : modules regarding network construction including codes for early stopping and batch generation
utils.py : other necessary modules
