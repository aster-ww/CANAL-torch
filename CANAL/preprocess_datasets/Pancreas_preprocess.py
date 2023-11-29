import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
from scEMAIL_preprocess import *
import csv
panglao = sc.read_h5ad('/data/wanh/CANAL/data/panglao_10000.h5ad')

data_path_human= "/data/wanh/CANAL/data/Pancreas/ALIGNED_Homo_sapiens_Pancreas.h5ad"

adata_human = sc.read(data_path_human)
adata_human.obs["organism"]='human'
print(adata_human,adata_human.var_names)
data = adata_human
print(data,data.var)

print(data,data.var,data.X)
#

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


print(data)
print(data.X)
new = ad.AnnData(X=counts)
new.var_names = ref
new.obs_names = data.obs_names
new.obs = data.obs
new.obs['celltype']=new.obs['cell_ontology_class']
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
print(new)#.X)
experiments = 'Pancreas'
with open('/data/wanh/CANAL/data/{}/{}_highly_gene_idx.csv'.format(experiments, experiments), 'w') as f:
    writer = csv.writer(f)
    for row in highly_variable_idx:
        writer.writerow([row])


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