import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import csv

def gene_align(adata, obj):
    """
    Conduct gene alignment between our data and panglaoDB dataset

    :adata (AnnData): the data we used to fine-tune the model or needed to be annotated
    :obj (Array): gene short name of this dataset

    :return: an AnnData Object, the genes of which is common with the PanglaoDB dataset. The element of genes which is absent in the orginal dataset will be zero.
    """
    panglao = sc.read_h5ad('/data/wanh/CANAL/data/panglao_10000.h5ad')
    counts = sparse.lil_matrix((adata.X.shape[0],panglao.X.shape[1]),dtype=np.float32)## sample_size * 16906
    ref = panglao.var_names.tolist()
    # obj = adata.var_names.tolist()#data.var['gene_short_name'].tolist()#

    overlap_genes = 0
    for i in range(len(ref)):
        if ref[i] in obj:
            # print("common gene:", ref[i])
            overlap_genes += 1
            loc = obj.index(ref[i])
            counts[:,i] = adata.X[:,loc]
    print("number of overlapping genes:",overlap_genes)
    counts = counts.tocsr()
    new = ad.AnnData(X=counts)
    new.var_names = ref
    new.obs_names = adata.obs_names
    new.obs = adata.obs
    new.obs['celltype']=new.obs['cell_ontology_class']
    new.uns = panglao.uns
    return new

def normalize(adata_aligned, experiments, dir, min_genes = 200, highly_gene_num = 1000):
    """
    Normalize the aligned scRNA-seq dataset, including quality control, log-normalization as well as highly variable gene selection.

    :adata_aligned (AnnData): the scRNA-seq dataset after gene alignment
    :experiments (str): name of experiments
    :dir (str): path to save the corresponding index of highly variable genes
    :min_genes (int): the minimum expressed gene number of a cell
    :highly_gene_num (int): preserved number of highly variable genes

    return: an AnnData object of the preprocessed scRNA-seq dataset
    """
    total_gene = adata_aligned.var.index
    sc.pp.filter_cells(adata_aligned, min_genes=min_genes)
    sc.pp.normalize_total(adata_aligned, target_sum=1e4)
    sc.pp.log1p(adata_aligned, base=2)
    sc.pp.highly_variable_genes(adata_aligned, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_gene_num, subset=True)
    highly_variable = np.array(adata_aligned.var.highly_variable.index)
    highly_variable_idx = []
    for i in range(len(total_gene)):
        if total_gene[i] in highly_variable:
            highly_variable_idx.append(int(i))
    highly_variable_idx = np.array(highly_variable_idx)
    with open('/{}/{}_highly_gene_idx.csv'.format(dir, experiments), 'w') as f:
        writer = csv.writer(f)
        for row in highly_variable_idx:
            writer.writerow([row])
    return adata_aligned

