import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import csv
experiments = 'Cross_tissue'
panglao = sc.read_h5ad('/data/wanh/CANAL/data/panglao_10000.h5ad')
data_path_10X = f"/data/wanh/CANAL/data/{experiments}/Quake_10x.h5ad"
data_path_SS2 = f"/data/wanh/CANAL/data/{experiments}/Quake_Smart-seq2.h5ad"
adata_10X = sc.read(data_path_10X)
adata_10X.obs["method"]='10X'
print(adata_10X,adata_10X.var)
adata_SS2 = sc.read(data_path_SS2)
adata_SS2.obs["method"]='SS2'
print(adata_SS2,adata_SS2.var)
data = adata_SS2.concatenate(adata_10X)
print(data,data.var)

preserve_idx = []
for i in range(data.X.shape[0]):
    if np.array(data.obs["organ"])[i] in ["Limb_Muscle", "Mammary_Gland", "Spleen", "Lung"]:
        preserve_idx.append(i)
data = data[preserve_idx]
print(data,data.var,data.X)


counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)## sample_size * 16906
ref = panglao.var_names.tolist()
# print("data.var_names:",data.var_names)
# print("data.var:",data.var)
obj = data.var_names.tolist()#data.var['gene_short_name'].tolist()#
obj = [i.upper() for i in obj]
overlap_genes = 0
for i in range(len(ref)):
    if ref[i] in obj:
        print(ref[i])
        # print("overlap_genes:", overlap_genes)
        overlap_genes += 1
        loc = obj.index(ref[i])
        counts[:,i] = data.X[:,loc]

print("overlap_genes:",overlap_genes)
counts = counts.tocsr()


print(data)
print(data.X)
new = ad.AnnData(X=counts)
new.var_names = ref
new.obs_names = data.obs_names
new.obs = data.obs
new.obs['celltype']=new.obs['cell_type1']
new.uns = panglao.uns

total_gene = new.var.index
highly_gene_num = 1000
print(new)
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
# new = batch_scale(new, chunk_size=20000, batch='batch')
print(new)

with open('/data/wanh/CANAL/data/{}/{}_highly_gene_idx.csv'.format(experiments, experiments), 'w') as f:
    writer = csv.writer(f)
    for row in highly_variable_idx:
        writer.writerow([row])


new = data.copy()
test_size = 500

np.random.seed(1)
organ1 = "Limb_Muscle"
batch1 = "10X"
data_subset1 = new[np.array(new.obs["method"]) == batch1]
data_subset1 = data_subset1[np.array(data_subset1.obs["organ"]) == organ1]
print("dataset 1:", batch1, organ1)
print(data_subset1)
print(np.unique(data_subset1.obs['cell_type1'],return_counts=True))
data_subset1.write("/data/wanh/CANAL/data/{}/{}_{}_{}.h5ad".format(experiments,"TabulaMuris",organ1,batch1))
shuffled_index_1 = np.random.permutation(len(data_subset1))
data1_train_idx = shuffled_index_1[test_size:]
data1_test_idx = shuffled_index_1[0:test_size]
data1_train= data_subset1[data1_train_idx]
data1_test= data_subset1[data1_test_idx]
print("data1_train:",np.unique(data1_train.obs.cell_type1,return_counts=True))
print("data1_test:",np.unique(data1_test.obs.cell_type1,return_counts=True))
data1_train.write("/data/wanh/CANAL/data/{}/{}_{}_{}_train.h5ad".format(experiments,"TabulaMuris",organ1,batch1))
data1_test.write('/data/wanh/CANAL/data/{}/{}_{}_{}_test.h5ad'.format(experiments,"TabulaMuris",organ1,batch1))

organ2 = "Mammary_Gland"
batch2 = "10X"
data_subset2 = new[np.array(new.obs["method"]) == batch2]
data_subset2 = data_subset2[np.array(data_subset2.obs["organ"]) == organ2]
print("dataset 2:", batch2, organ2)
print(data_subset2)
print(np.unique(data_subset2.obs['cell_type1'],return_counts=True))
data_subset2.write("/data/wanh/CANAL/data/{}/{}_{}_{}.h5ad".format(experiments,"TabulaMuris",organ2,batch2))
shuffled_index_2 = np.random.permutation(len(data_subset2))
data2_train_idx = shuffled_index_2[test_size:]
data2_test_idx = shuffled_index_2[0:test_size]
data2_train= data_subset2[data2_train_idx]
data2_test= data_subset2[data2_test_idx]
print("data2_train:",np.unique(data2_train.obs.cell_type1,return_counts=True))
print("data2_test:",np.unique(data2_test.obs.cell_type1,return_counts=True))
data2_train.write("/data/wanh/CANAL/data/{}/{}_{}_{}_train.h5ad".format(experiments,"TabulaMuris",organ2,batch2))
data2_test.write('/data/wanh/CANAL/data/{}/{}_{}_{}_test.h5ad'.format(experiments,"TabulaMuris",organ2,batch2))


organ3 = "Spleen"
batch3 = "10X"
preserve_celltype3=['B cell', 'T cell', 'macrophage', 'natural killer cell']
data_subset3 = new[np.array(new.obs["method"]) == batch3]
data_subset3 = data_subset3[np.array(data_subset3.obs["organ"]) == organ3]
data_subset3 = data_subset3[[data_subset3.obs["cell_type1"][i] in preserve_celltype3 for i in range(len(data_subset3))]]
print("dataset 3:", batch3, organ3)
print(data_subset3)
print(np.unique(data_subset3.obs['cell_type1'],return_counts=True))
data_subset3.write("/data/wanh/CANAL/data/{}/{}_{}_{}.h5ad".format(experiments,"TabulaMuris",organ3,batch3))
shuffled_index_3 = np.random.permutation(len(data_subset3))
data3_train_idx = shuffled_index_3[test_size:]
data3_test_idx = shuffled_index_3[0:test_size]
data3_train= data_subset3[data3_train_idx]
data3_test= data_subset3[data3_test_idx]
print("data3_train:",np.unique(data3_train.obs.cell_type1,return_counts=True))
print("data3_test:",np.unique(data3_test.obs.cell_type1,return_counts=True))
data3_train.write("/data/wanh/CANAL/data/{}/{}_{}_{}_train.h5ad".format(experiments,"TabulaMuris",organ3,batch3))
data3_test.write('/data/wanh/CANAL/data/{}/{}_{}_{}_test.h5ad'.format(experiments,"TabulaMuris",organ3,batch3))

organ4 = "Lung"
batch4 = "10X"
preserve_celltype4=['B cell', 'T cell', 'alveolar macrophage', 'lung endothelial cell', 'natural killer cell','stromal cell']
data_subset4 = new[np.array(new.obs["method"]) == batch4]
data_subset4 = data_subset4[np.array(data_subset4.obs["organ"]) == organ4]
data_subset4 = data_subset4[[data_subset4.obs["cell_type1"][i] in preserve_celltype4 for i in range(len(data_subset4))]]
print("dataset 4:", batch4, organ4)
print(data_subset4)
print(np.unique(data_subset4.obs['cell_type1'],return_counts=True))
data_subset4.write("/data/wanh/CANAL/data/{}/{}_{}_{}.h5ad".format(experiments,"TabulaMuris",organ4,batch4))
shuffled_index_4 = np.random.permutation(len(data_subset4))
data4_train_idx = shuffled_index_4[test_size:]
data4_test_idx = shuffled_index_4[0:test_size]
data4_train= data_subset4[data4_train_idx]
data4_test= data_subset4[data4_test_idx]
print("data4_train:",np.unique(data4_train.obs.cell_type1,return_counts=True))
print("data4_test:",np.unique(data4_test.obs.cell_type1,return_counts=True))
data4_train.write("/data/wanh/CANAL/data/{}/{}_{}_{}_train.h5ad".format(experiments,"TabulaMuris",organ4,batch4))
data4_test.write('/data/wanh/CANAL/data/{}/{}_{}_{}_test.h5ad'.format(experiments,"TabulaMuris",organ4,batch4))

organ5 = "Limb_Muscle"
batch5 = "SS2"
data_subset5 = new[np.array(new.obs["method"]) == batch5]
data_subset5 = data_subset5[np.array(data_subset5.obs["organ"]) == organ5]
print("dataset 5:", batch5, organ5)
print(data_subset5)
print(np.unique(data_subset5.obs['cell_type1'],return_counts=True))
data_subset5.write("/data/wanh/CANAL/data/{}/{}_{}_{}.h5ad".format(experiments,"TabulaMuris",organ5,batch5))

organ6 = "Mammary_Gland"
batch6 = "SS2"
data_subset6 = new[np.array(new.obs["method"]) == batch6]
data_subset6 = data_subset6[np.array(data_subset6.obs["organ"]) == organ6]
print("dataset 6:", batch6, organ6)
print(data_subset6)
print(np.unique(data_subset6.obs['cell_type1'],return_counts=True))
data_subset6.write("/data/wanh/CANAL/data/{}/{}_{}_{}.h5ad".format(experiments,"TabulaMuris",organ6,batch6))

organ7 = "Spleen"
batch7 = "SS2"
data_subset7 = new[np.array(new.obs["method"]) == batch7]
data_subset7 = data_subset7[np.array(data_subset7.obs["organ"]) == organ7]
print("dataset 7:", batch7, organ7)
print(data_subset7)
print(np.unique(data_subset7.obs['cell_type1'],return_counts=True))
data_subset7.write("/data/wanh/CANAL/data/{}/{}_{}_{}.h5ad".format(experiments,"TabulaMuris",organ7,batch7))

organ8 = "Lung"
batch8 = "SS2"
data_subset8 = new[np.array(new.obs["method"]) == batch8]
data_subset8 = data_subset8[np.array(data_subset8.obs["organ"]) == organ8]
data_subset8 = data_subset8[[data_subset8.obs["cell_type1"][i] in preserve_celltype4 for i in range(len(data_subset8))]]
print("dataset 8:", batch8, organ8)
print(data_subset8)
print(np.unique(data_subset8.obs['cell_type1'],return_counts=True))
data_subset8.write("/data/wanh/CANAL/data/{}/{}_{}_{}.h5ad".format(experiments,"TabulaMuris",organ8,batch8))
