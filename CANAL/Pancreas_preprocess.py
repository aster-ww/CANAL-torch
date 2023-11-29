import os
os.chdir('/data/wanh/CANAL/')
import sys
sys.path.append('/data/wanh/CANAL/')
from preprocess import *
import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import csv


data_path_human= "/data/wanh/CANAL/data/Pancreas/ALIGNED_Homo_sapiens_Pancreas.h5ad"
adata_human = sc.read(data_path_human)
adata_human.obs["organism"]='human'
print(adata_human,adata_human.var_names)
data = adata_human
print(data,data.var)
obj = data.var_names.tolist()
new_data = gene_align(data,obj)
print(new_data)

new = normalize(new_data, "Pancreas",dir="/data/wanh/CANAL/data/Pancreas")

dataset1 = "Muraro"
data_subset1 = new[np.array(new.obs["dataset_name"]) == dataset1]
print("dataset 1:", dataset1)
print(data_subset1)
print(np.unique(data_subset1.obs['cell_ontology_class'],return_counts=True))
# data_subset1.write("/data/wanh/CANAL/data/{}/{}.h5ad".format(experiments,dataset1))

dataset2 = "Enge"
data_subset2 = new[np.array(new.obs["dataset_name"]) == dataset2]
print("dataset 2:", dataset2)
print(data_subset2)
print(np.unique(data_subset2.obs['cell_ontology_class'],return_counts=True))
# data_subset2.write("/data/wanh/CANAL/data/{}/{}.h5ad".format(experiments,dataset2))

dataset3 = "Baron_human"
data_subset3 = new[np.array(new.obs["dataset_name"]) == dataset3]
print("dataset 3:", dataset3)
print(data_subset3)
print(np.unique(data_subset3.obs['cell_ontology_class'],return_counts=True))
# data_subset3.write("/data/wanh/CANAL/data/{}/{}.h5ad".format(experiments,dataset3))

dataset4 = "Segerstolpe"
data_subset4 = new[np.array(new.obs["dataset_name"]) == dataset4]
print("dataset 4:", dataset4)
print(data_subset4)
print(np.unique(data_subset4.obs['cell_ontology_class'],return_counts=True))
# data_subset4.write("/data/wanh/CANAL/data/{}/{}.h5ad".format(experiments,dataset4))



preserve_celltype1 = ['acinar', 'alpha', 'beta', 'delta', 'ductal',  'gamma', 'mesenchymal']
data_subset1 = data_subset1[[data_subset1.obs.cell_type1[i] in preserve_celltype1 for i in range(len(data_subset1))]]
test_size = 500
np.random.seed(1)
shuffled_index_1 = np.random.permutation(len(data_subset1))
data_subset1_train_idx = shuffled_index_1[test_size:]
data_subset1_test_idx = shuffled_index_1[0:test_size]
data_subset1_train= data_subset1[data_subset1_train_idx]
data_subset1_test= data_subset1[data_subset1_test_idx]
print("data_subset1_train:",np.unique(data_subset1_train.obs.cell_type1,return_counts=True))
print("data_subset1_test:",np.unique(data_subset1_test.obs.cell_type1,return_counts=True))
data_subset1_train.write('/data/wanh/CANAL/data/{}/{}_train.h5ad'.format(experiments, dataset1))
data_subset1_test.write('/data/wanh/CANAL/data/{}/{}_test.h5ad'.format(experiments, dataset1))


np.random.seed(2)
shuffled_index_2 = np.random.permutation(len(data_subset2))
data_subset2_train_idx = shuffled_index_2[test_size:]
data_subset2_test_idx = shuffled_index_2[0:test_size]
data_subset2_train= data_subset2[data_subset2_train_idx]
data_subset2_test= data_subset2[data_subset2_test_idx]
print("data_subset2_train:",np.unique(data_subset2_train.obs.cell_type1,return_counts=True))
print("data_subset2_test:",np.unique(data_subset2_test.obs.cell_type1,return_counts=True))
data_subset2_train.write('/data/wanh/CANAL/data/{}/{}_train.h5ad'.format(experiments, dataset2))
data_subset2_test.write('/data/wanh/CANAL/data/{}/{}_test.h5ad'.format(experiments, dataset2))


preserve_celltype3 = ['acinar', 'activated_stellate', 'alpha', 'beta', 'delta', 'ductal',
                      'endothelial', 'gamma', 'quiescent_stellate']
data_subset3 = data_subset3[[data_subset3.obs.cell_type1[i] in preserve_celltype3 for i in range(len(data_subset3))]]
np.random.seed(3)
shuffled_index_3 = np.random.permutation(len(data_subset3))
data_subset3_train_idx = shuffled_index_3[test_size:]
data_subset3_test_idx = shuffled_index_3[0:test_size]
data_subset3_train= data_subset3[data_subset3_train_idx]
data_subset3_test= data_subset3[data_subset3_test_idx]
print("data_subset3_train:",np.unique(data_subset3_train.obs.cell_type1,return_counts=True))
print("data_subset3_test:",np.unique(data_subset3_test.obs.cell_type1,return_counts=True))
data_subset3_train.write('/data/wanh/CANAL/data/{}/{}_train.h5ad'.format(experiments, dataset3))
data_subset3_test.write('/data/wanh/CANAL/data/{}/{}_test.h5ad'.format(experiments, dataset3))

data_train=sc.AnnData.concatenate(data_subset1_train,data_subset2_train,data_subset3_train,batch_key='batch2')
print("data_train:",np.unique(data_train.obs.cell_type1,return_counts=True))
# data_train.write('/data/wanh/CANAL/data/{}/{}_train.h5ad'.format(experiments, "Pancreas"))


preserve_celltype4 = np.unique(data_train.obs.cell_type1)
data_subset4 = data_subset4[[data_subset4.obs.cell_type1[i] in preserve_celltype4 for i in range(len(data_subset4))]]
np.random.seed(4)
shuffled_index_4 = np.random.permutation(len(data_subset4))
data_subset4_test_idx = shuffled_index_4[0:test_size]
data_subset4_test= data_subset4[data_subset4_test_idx]
print("data_subset4_test:",np.unique(data_subset4_test.obs.cell_type1,return_counts=True))

data_test = sc.AnnData.concatenate(data_subset1_test,data_subset2_test,data_subset3_test,data_subset4_test,batch_key='batch2')
print("data_test:",np.unique(data_test.obs.cell_type1,return_counts=True))
data_test.write('/data/wanh/CANAL/data/{}/{}_test.h5ad'.format(experiments, "Pancreas"))