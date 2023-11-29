import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
from scEMAIL_preprocess import *
import csv
panglao = sc.read_h5ad('/data/wanh/CANAL/data/panglao_10000.h5ad')
experiments = "Human_immune"
data = sc.read_h5ad('/data/wanh/CANAL/data/{}.h5ad'.format(experiments))
data.X = data.layers["counts"].copy()
data.X = data.X.astype(int)
print(data)
print(data.X)
counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)## sample_size * 16906
ref = panglao.var_names.tolist()
obj = data.var_names.tolist()#data.var['gene_short_name'].tolist()#
overlap_genes = 0
for i in range(len(ref)):
    if ref[i] in obj:
        print(ref[i])
        print("overlap_genes:", overlap_genes)
        overlap_genes += 1
        loc = obj.index(ref[i])
        counts[:,i] = data.X[:,loc]

print("overlap_genes:",overlap_genes)
counts = counts.tocsr()
new = ad.AnnData(X=counts)
new.var_names = ref
new.obs_names = data.obs_names
new.obs = data.obs
new.obs['celltype']=new.obs['final_annotation']
new.uns = panglao.uns

total_gene = new.var.index
highly_gene_num = 1000
print(new)#.X)
sc.pp.filter_cells(new, min_genes=200)
sc.pp.normalize_total(new, target_sum=1e4)
sc.pp.log1p(new, base=2)
sc.pp.highly_variable_genes(new, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_gene_num, subset=True)
highly_variable = np.array(new.var.highly_variable.index)
highly_variable_idx = []
for i in range(len(total_gene)):
    if total_gene[i] in highly_variable:
        highly_variable_idx.append(int(i))
highly_variable_idx = np.array(highly_variable_idx)
print(highly_variable_idx)
print(new)#.X)
with open('./data/{}/{}_highly_gene_idx.csv'.format(experiments,experiments), 'w') as f:
    writer = csv.writer(f)
    for row in highly_variable_idx:
        writer.writerow([row])


stage_num = 3

for i in range(stage_num + 2):
    if i == 0:
        batch1 = "10X"
        data_subset1 = data[[batch1 in i for i in np.array(data.obs["batch"])]]
        print("stage 1:", batch1)
        print(data_subset1)
        subset1_celltype = np.unique(data_subset1.obs['final_annotation'])
        print(np.unique(data_subset1.obs['final_annotation'],return_counts=True))
        # data_subset1.write("./data/{}/{}_total_{}_train.h5ad".format(experiments, experiments, stage_num))
    elif i == 1:
        batch2 = "Oetjen"
        data_subset2 = data[[batch2 in i for i in np.array(data.obs["batch"])]]
        print("stage 2:", batch2)
        print(data_subset2)
        subset2_celltype = np.unique(data_subset2.obs['final_annotation'])
        print(np.unique(data_subset2.obs['final_annotation'],return_counts=True))
        # data_subset2.write("./data/{}/{}_total_{}_train.h5ad".format(experiments, experiments, stage_num))
    elif i == 2:
        batch3 = "Sun"
        data_subset3 = data[[batch3 in i for i in np.array(data.obs["batch"])]]
        print("stage 3:", batch3)
        print(data_subset3)
        subset3_celltype = np.unique(data_subset3.obs['final_annotation'])
        print(np.unique(data_subset3.obs['final_annotation'],return_counts=True))
        # data_subset3.write("./data/{}/{}_total_{}_train.h5ad".format(experiments, experiments, stage_num))
    elif i == 3:
        batch4 = "Freytag"
        data_subset4 = data[[batch4 in i for i in np.array(data.obs["batch"])]]
        print("stage 4:", batch4)
        print(data_subset4)
        subset4_celltype = np.unique(data_subset4.obs['final_annotation'])
        print(np.unique(data_subset4.obs['final_annotation'],return_counts=True))
        # data_subset4.write("./data/{}/{}_total_{}_test.h5ad".format(experiments, experiments, stage_num))
    else:
        batch5 = "Villani"
        data_subset5 = data[[batch5 in i for i in np.array(data.obs["batch"])]]
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

