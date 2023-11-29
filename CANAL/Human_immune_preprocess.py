import os
os.chdir('/data/wanh/CANAL/')
import sys
sys.path.append('/data/wanh/CANAL/')
from preprocess import *
import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import csv



experiments = "Human_immune"
data = sc.read_h5ad('/data/wanh/CANAL/data/{}.h5ad'.format(experiments))
data.X = data.layers["counts"].copy()
data.X = data.X.astype(int)
print(data)
print(data.X)
obj = data.var_names.tolist()#data.var['gene_short_name'].tolist()#
new_data = gene_align(data,obj)
print(new_data)
new = normalize(new_data, experiments,dir=f"/data/wanh/CANAL/data/{experiments}")


stage_num = 3

for i in range(stage_num + 2):
    if i == 0:
        batch1 = "10X"
        data_subset1 = new[[batch1 in i for i in np.array(new.obs["batch"])]]
        print("stage 1:", batch1)
        print(data_subset1)
        subset1_celltype = np.unique(data_subset1.obs['final_annotation'])
        print(np.unique(data_subset1.obs['final_annotation'],return_counts=True))
        # data_subset1.write("./data/{}/{}_total_{}_train.h5ad".format(experiments, experiments, stage_num))
    elif i == 1:
        batch2 = "Oetjen"
        data_subset2 = new[[batch2 in i for i in np.array(new.obs["batch"])]]
        print("stage 2:", batch2)
        print(data_subset2)
        subset2_celltype = np.unique(data_subset2.obs['final_annotation'])
        print(np.unique(data_subset2.obs['final_annotation'],return_counts=True))
        # data_subset2.write("./data/{}/{}_total_{}_train.h5ad".format(experiments, experiments, stage_num))
    elif i == 2:
        batch3 = "Sun"
        data_subset3 = new[[batch3 in i for i in np.array(new.obs["batch"])]]
        print("stage 3:", batch3)
        print(data_subset3)
        subset3_celltype = np.unique(data_subset3.obs['final_annotation'])
        print(np.unique(data_subset3.obs['final_annotation'],return_counts=True))
        # data_subset3.write("./data/{}/{}_total_{}_train.h5ad".format(experiments, experiments, stage_num))
    elif i == 3:
        batch4 = "Freytag"
        data_subset4 = new[[batch4 in i for i in np.array(new.obs["batch"])]]
        print("stage 4:", batch4)
        print(data_subset4)
        subset4_celltype = np.unique(data_subset4.obs['final_annotation'])
        print(np.unique(data_subset4.obs['final_annotation'],return_counts=True))
        # data_subset4.write("./data/{}/{}_total_{}_test.h5ad".format(experiments, experiments, stage_num))
    else:
        batch5 = "Villani"
        data_subset5 = new[[batch5 in i for i in np.array(new.obs["batch"])]]
        print("stage 5:", batch5)
        print(data_subset5)
        subset5_celltype = np.unique(data_subset5.obs['final_annotation'])
        print(np.unique(data_subset5.obs['final_annotation'], return_counts=True))
        # data_subset4.write("./data/{}/{}_total_{}_test.h5ad".format(experiments, experiments, stage_num))



common_cell_type=[i for i in subset1_celltype if i in subset2_celltype and i in subset3_celltype]
print("common cell type:",common_cell_type)
data_subset1_closed = data_subset1[[i in common_cell_type for i in np.array(data_subset1.obs['final_annotation'])]]
data_subset2_closed = data_subset2[[i in common_cell_type for i in np.array(data_subset2.obs['final_annotation'])]]
data_subset3_closed = data_subset3[[i in common_cell_type for i in np.array(data_subset3.obs['final_annotation'])]]
data_subset4_closed = data_subset4[[i in common_cell_type for i in np.array(data_subset4.obs['final_annotation'])]]
data_subset5_closed = data_subset5[[i in common_cell_type for i in np.array(data_subset5.obs['final_annotation'])]]

test_size = 1000
np.random.seed(1111)
shuffled_index_1 = np.random.permutation(len(data_subset1_closed))

data_subset1_closed_train_idx = shuffled_index_1[test_size:]
data_subset1_closed_test_idx = shuffled_index_1[0:test_size]
data_subset1_closed_train= data_subset1_closed[data_subset1_closed_train_idx]
data_subset1_closed_test= data_subset1_closed[data_subset1_closed_test_idx]
print("final stage 1 train:",data_subset1_closed_train)
print("cell types:",len(np.unique(data_subset1_closed_train.obs['final_annotation'])))
print("\n")
print("final stage 1 test:",data_subset1_closed_test)
print("cell types:",len(np.unique(data_subset1_closed_test.obs['final_annotation'])))
print("\n\n\n")
# print(np.unique(data_subset1_closed.obs['final_annotation']))
data_subset1_closed_train.write("./data/{}/{}_stage_{}_total_{}_train.h5ad".format(experiments,experiments,3,stage_num))
data_subset1_closed_test.write("./data/{}/{}_stage_{}_total_{}_test.h5ad".format(experiments,experiments,3,stage_num))

np.random.seed(2222)
shuffled_index_2 = np.random.permutation(len(data_subset2_closed))

data_subset2_closed_train_idx = shuffled_index_2[test_size:]
data_subset2_closed_test_idx = shuffled_index_2[0:test_size]
data_subset2_closed_train = data_subset2_closed[data_subset2_closed_train_idx]
data_subset2_closed_test = data_subset2_closed[data_subset2_closed_test_idx]
print("final stage 2 train:",data_subset2_closed_train)
print("cell types:",len(np.unique(data_subset2_closed_train.obs['final_annotation'])))
print("\n")
print("final stage 2 test:",data_subset2_closed_test)
print("cell types:",len(np.unique(data_subset2_closed_test.obs['final_annotation'])))
print("\n\n\n")
# print(np.unique(data_subset1_closed.obs['final_annotation']))
data_subset2_closed_train.write("./data/{}/{}_stage_{}_total_{}_train.h5ad".format(experiments,experiments,1,stage_num))
data_subset2_closed_test.write("./data/{}/{}_stage_{}_total_{}_test.h5ad".format(experiments,experiments,1,stage_num))

np.random.seed(3333)
shuffled_index_3 = np.random.permutation(len(data_subset3_closed))

data_subset3_closed_train_idx = shuffled_index_3[test_size:]
data_subset3_closed_test_idx = shuffled_index_3[0:test_size]
data_subset3_closed_train = data_subset3_closed[data_subset3_closed_train_idx]
data_subset3_closed_test = data_subset3_closed[data_subset3_closed_test_idx]
print("final stage 3 train:",data_subset3_closed_train)
print("cell types:",len(np.unique(data_subset3_closed_train.obs['final_annotation'])))
print("\n")
print("final stage 3 test:",data_subset3_closed_test)
print("cell types:",len(np.unique(data_subset3_closed_test.obs['final_annotation'])))
print("\n\n\n")
# print(np.unique(data_subset1_closed.obs['final_annotation']))
data_subset3_closed_train.write("./data/{}/{}_stage_{}_total_{}_train.h5ad".format(experiments,experiments,2,stage_num))
data_subset3_closed_test.write("./data/{}/{}_stage_{}_total_{}_test.h5ad".format(experiments,experiments,2,stage_num))

offline_data_closed_train = data_subset1_closed_train.concatenate(data_subset2_closed_train).concatenate(data_subset3_closed_train)
print("offline dataset:",offline_data_closed_train)
offline_data_closed_train.write("./data/{}/{}_stage_{}_total_{}_train.h5ad".format(experiments,experiments,1,1))

print("final stage 4:",data_subset4_closed)
print(np.unique(data_subset4_closed.obs['final_annotation']))
data_subset4_closed.write("./data/{}/{}_stage_{}_total_{}_test.h5ad".format(experiments,experiments,4,stage_num))

print("final stage 5:",data_subset5_closed)
print(np.unique(data_subset5_closed.obs['final_annotation']))
data_subset5_closed.write("./data/{}/{}_stage_{}_total_{}_test.h5ad".format(experiments,experiments,5,stage_num))

