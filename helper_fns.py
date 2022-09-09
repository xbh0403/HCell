from multiprocessing.dummy import Array
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import math
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns
import biomart
from sklearn.metrics import confusion_matrix


# Train test split & Cross-validation
def costumized_train_test_split(dataset, cross_validation=False, obs_key='Manually_curated_celltype', k_fold=5):
    indices_by_celltypes = {}
    train_indices, test_indices, cv = [], [], []
    for cell_type in dataset.obs[obs_key].unique():
        indices = np.where(dataset.obs[obs_key] == cell_type)[0]
        np.random.shuffle(indices, random_state=1)
        indices_by_celltypes.update({cell_type: indices})
        split = int(len(indices)/k_fold)
        if cross_validation:
            for i in range(k_fold):
                temp = i*split
                temp_test = list(indices[temp:temp+split])
                temp_train = list(set(indices) - set(temp_test))
                if cell_type != dataset.obs[obs_key].unique()[0]:
                    cv[i].get("train").extend(temp_train)
                    cv[i].get("test").extend(temp_test)
                else:
                    cv.append({"train":temp_train, "test": temp_test})
        else:
            test_indices.extend(indices[:split])
            train_indices.extend(indices[split:])
    return train_indices, test_indices, cv


# Feature Selection by Scanpy
def select_features(dataset_training, num_genes, list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner):
    print("feature_selection")
    dataset_training.var['mt'] = dataset_training.var_names.str.startswith('MT-', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(dataset_training, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc_pp_train = sc.pp.filter_cells(dataset_training, min_genes=200, copy=True)
    sc.pp.filter_genes(sc_pp_train, min_cells=3)
    sc_pp_train = sc_pp_train[sc_pp_train.obs.n_genes_by_counts < 2500, :]
    sc_pp_train = sc_pp_train[sc_pp_train.obs.pct_counts_mt < 5, :]
    sc.pp.highly_variable_genes(sc_pp_train, n_top_genes=int(num_genes/4))
    sc_pp_train = sc_pp_train[:, sc_pp_train.var.highly_variable]
    return sc_pp_train

# Preprocessing by PCA
def prepPCA(dataset_training, num_genes=36601):
    sc_pp_train = select_features(dataset_training, num_genes)
    sc_pp_train = sc.pp.normalize_total(sc_pp_train, target_sum=1)
    sc_pp_train = sc.pp.log1p(sc_pp_train)
    sc_pp_train = sc.pp.pca(sc_pp_train, n_comps=int(num_genes/4))
    return sc_pp_train


def log_likelihood_Gaussian(x, mu, log_var):
    if torch.cuda.is_available():
        log_likelihood = torch.sum(-0.5 * torch.log(2*torch.tensor(math.pi).cuda()) - 0.5 * log_var - 0.5 * (x - mu)**2 / torch.exp(log_var), dim=1)
    else:
        log_likelihood = torch.sum(-0.5 * torch.log(2*torch.tensor(math.pi)) - 0.5 * log_var - 0.5 * (x - mu)**2 / torch.exp(log_var), dim=1)
    return log_likelihood


def log_likelihood_student(x, mu, log_var, df=2.0):
    # df=mu.shape[0]-1
    sigma = torch.sqrt(torch.exp(log_var))

    dist = torch.distributions.StudentT(df=df,
                                        loc=mu,
                                        scale=sigma)
    return torch.sum(dist.log_prob(x), dim=1)


def transform(x, list_ct ,list_inner_nodes, encoder_celltype, encoder_celltype_inner):
    if x in list_inner_nodes:
        return encoder_celltype_inner.transform([x])[0] + len(list_ct)
    else:
        return encoder_celltype.transform([x])[0]


def build_hierarchical_tree_celltypist(all_nodes, list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner):
    g = ig.Graph()
    g.add_vertices(len(all_nodes))
    g.vs['name'] = np.append(encoder_celltype.inverse_transform(list(range(len(list_ct)))), encoder_celltype_inner.inverse_transform(list(range(len(list_inner_nodes)))))
    g.add_edges([(transform('Cross-tissue Immune Cell Atlas', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Cross-tissue Immune Cell Atlas', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Myeloid', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Cross-tissue Immune Cell Atlas', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('T & Innate lymphoid cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('ABCs', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Germinal center B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Memory B cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Naive B cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Plasma cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Plasmablasts', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Pre-B', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Pro-B', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('Germinal center B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('GC_B (I)', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Germinal center B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('GC_B (II)', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('Myeloid', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Cycling', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Myeloid', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Myeloid', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Macrophages', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Myeloid', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Monocytes', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('Dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('DC1', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('DC2', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('migDC', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('Macrophages', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Alveolar macrophages', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Macrophages', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Erythrophagocytic macrophages', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Macrophages', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Intermediate macrophages', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Macrophages', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Intestinal macrophages', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('Monocytes', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Classical monocytes', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Monocytes', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Nonclassical monocytes', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('T & Innate lymphoid cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T & Innate lymphoid cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T & Innate lymphoid cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Cycling T&NK', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T & Innate lymphoid cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('ILC3', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T & Innate lymphoid cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('NK', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T & Innate lymphoid cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('T_CD4/CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('T Naive', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Teffector/EM_CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Tfh', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Tregs', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Trm_Th1/Th17', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('MAIT', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Tem/emra_CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Tgd_CRTAM+', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Tissue-resident memory T (Trm) cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Tnaive/CM_CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Trm_Tgd', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('NK', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('NK_CD16+', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('NK', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('NK_CD56bright_CD16-', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('T Naive', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Tnaive/CM_CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T Naive', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Tnaive/CM_CD4_activated', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    g.add_edges([(transform('Tissue-resident memory T (Trm) cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Trm/em_CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('Tissue-resident memory T (Trm) cells', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Trm_gut_CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))])
    return g


def build_hierarchical_tree_popv_immune(all_nodes, list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner):
    g = ig.Graph()
    g.add_vertices(len(all_nodes))
    g.vs['name'] = np.append(encoder_celltype.inverse_transform(list(range(len(list_ct)))), encoder_celltype_inner.inverse_transform(list(range(len(list_inner_nodes)))))
    g.add_edges([(transform('popv_immune', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('mesenchymal stem cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('popv_immune', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('hematopoietic stem cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('popv_immune', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('popv_immune', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('common myeloid progenitor', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('common myeloid progenitor', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('myeloid cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('myeloid cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('platelet', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('myeloid cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('erythroid lineage cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('myeloid cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('myeloid leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                
                (transform('erythroid lineage cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('erythroid progenitor cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('erythroid lineage cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('erythrocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('myeloid leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('monocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('myeloid leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('mast cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('myeloid leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('macrophage', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('myeloid leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner),transform('granulocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('myeloid leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('myeloid dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('monocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('classical monocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('monocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('intermediate monocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('monocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('non-classical monocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('macrophage', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('microglial cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('granulocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('basophil', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('granulocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('neutrophil', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('myeloid dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD1c-positive myeloid dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('myeloid dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD141-positive myeloid dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('innate lymphoid cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('leukocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('liver dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('mature conventional dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('plasmacytoid dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('Langerhans cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('mature conventional dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('myeloid dendritic cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('mature B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('plasma cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('plasmablast', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                
                (transform('mature B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('naive B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('mature B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('memory B cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('regulatory T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('NK', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('CD8', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD8-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD8-positive, alpha-beta cytokine secreting effector T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('effector CD8-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD8-positive, alpha-beta memory T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD8-positive, alpha-beta cytotoxic T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD8-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('naive thymus-derived CD8-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('CD4', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD4-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD4-positive helper T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('naive thymus-derived CD4-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('CD4-positive, alpha-beta memory T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('effector CD4-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('T follicular helper cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('CD4-positive, alpha-beta T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('naive regulatory T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('regulatory T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('naive regulatory T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                # Commented out because there are only 6 cells in the dataset
                # (transform('thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('double-positive, alpha-beta thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('DN3 thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('DN4 thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('DN1 thymic pro-T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),

                (transform('NK', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('mature NK T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('NK', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('immature natural killer cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
                (transform('NK', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('type I NK T cell', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner))
                ])
    return g


def plot_hierarchical_tree(g):
    layout = g.layout("kamada_kawai")
    ig.plot(g, layout=layout, vertex_label=g.vs["name"], vertex_label_size=10, vertex_size=15, bbox=(1000, 1000), margin=100, vertex_color='white')


def get_shortest_dist(node_1, node_2, graph):
    return len(graph.get_shortest_paths(node_1, node_2)[0])-1


def get_dist_df(list_num_ct, g):
    dist_df = pd.DataFrame(0, index=np.arange(len(list_num_ct)), columns=np.arange(len(list_num_ct)))
    for i in range(len(list_num_ct)):
        for j in range(len(list_num_ct)):
            dist_df.iloc[i, j]=get_shortest_dist(i, j, g)
    return dist_df


def get_ensembl_mappings():                                   
    # Set up connection to server
    # server = biomart.BiomartServer('http://uswest.ensembl.org/biomart')
    server = biomart.BiomartServer('http://ensembl.org/biomart')
    mart = server.datasets['hsapiens_gene_ensembl']
    attributes = ['hgnc_symbol', 'ensembl_gene_id']
                                                                                
    # Get the mapping between the attributes
    response = mart.search({'attributes': attributes})
    data = response.raw.data.decode('ascii')
                                                                                
    ensembl_to_genesymbol = {}
    # Store the data in a dict                                                  
    for line in data.splitlines():
        line = line.split('\t')
        hgnc_symbol = line[0]
        ensembl_gene_id = line[1]
        ensembl_to_genesymbol[hgnc_symbol] = ensembl_gene_id
         
    return ensembl_to_genesymbol

def plot_distance_matrix(mode, model, dist_df, dataset, encoder_celltype, test_indices):
    y_test = encoder_celltype.transform(dataset[test_indices].obs['Manually_curated_celltype'])
    if mode == 'Net':
        if torch.cuda.is_available():
            y_pred = model(torch.tensor(dataset[test_indices].X).cuda())
        else:
            y_pred = model(torch.tensor(dataset[test_indices].X))
    elif mode == 'Proto_Net':
        if torch.cuda.is_available():
            y_pred, y_embeddings = model(torch.tensor(dataset[test_indices].X).cuda())
        else:
            y_pred, y_embeddings = model(torch.tensor(dataset[test_indices].X))
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.argmax(axis=1)
    dist_list = []
    for i in range(len(y_pred)):
        distance = dist_df.iloc[y_pred[i], y_test[i]]
        dist_list.append(distance)
    print(np.mean(np.array(dist_list)))
    sns.displot(dist_list)
    plt.show(block=False)
    return np.mean(np.array(dist_list))


def plot_confusion_matrix(mode, model, dataset, encoder, test_indices, obs_name, size=30):
    y_test = dataset[test_indices].obs[obs_name]
    if mode == 'Net':
        if torch.cuda.is_available():
            y_pred = model(torch.tensor(dataset[test_indices].X).cuda())
        else:
            y_pred = model(torch.tensor(dataset[test_indices].X))
    elif mode == 'Proto_Net':
        if torch.cuda.is_available():
            y_pred, y_embeddings = model(torch.tensor(dataset[test_indices].X).cuda())
        else:
            y_pred, y_embeddings = model(torch.tensor(dataset[test_indices].X))
    y_pred = y_pred.detach().cpu().numpy()
    pred = encoder.inverse_transform(y_pred.argmax(axis=1))
    cm = confusion_matrix(y_test, pred)
    # Normalise
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(size,size))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=encoder.inverse_transform(range(len(dataset.obs[obs_name].unique().tolist()))),
                                        yticklabels=encoder.inverse_transform(range(len(dataset.obs[obs_name].unique().tolist()))))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show(block=False)


# Get number of cells by cell type
def get_num_by_ct(label, dataset, obs):
    return len(dataset.obs[dataset.obs[obs] == label])


def get_weights(num_celltypes, encoder, dataset, obs):
    weights = []
    for i in range(num_celltypes):
        weights.append(get_num_by_ct(encoder.inverse_transform([i])[0], dataset, obs))
    weights = torch.tensor(weights, dtype=float)
    return weights


def get_embeddings_and_out(model, dataset, encoder):
    if torch.cuda.is_available():
        y_pred, y_embeddings = model(torch.tensor(dataset.X).cuda())
    else:
        y_pred, y_embeddings = model(torch.tensor(dataset.X))
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = encoder.inverse_transform(y_pred.argmax(axis=1))
    y_embeddings = y_embeddings.detach().cpu().numpy()
    return y_embeddings, y_pred


def get_prototypes_and_labels(model, encoder, num_celltypes):
    embedding_prototypes = model.prototypes.detach().cpu().numpy()
    embedding_prototypes_labels = encoder.inverse_transform(range(num_celltypes))
    return embedding_prototypes, embedding_prototypes_labels


def print_true_positive_given_labels_and_datasets(test_true_labels, test_pred_labels, train_celltype, test_celltype):
    cell_count = 0
    cell_pred = 0
    for i in range(len(test_pred_labels)):
        if test_true_labels[i] == test_celltype:
            cell_count += 1
            if test_pred_labels[i] == train_celltype:
                cell_pred += 1
    print("True positive " + test_celltype + ": " + str(cell_pred/cell_count * 100) + '%')

    wrong_dict = {}
    for i in range(len(test_true_labels)):
        if test_true_labels[i] == test_celltype and test_pred_labels[i] != train_celltype:
            if test_pred_labels[i] in wrong_dict.keys():
                wrong_dict[test_pred_labels[i]] += 1
            else:
                wrong_dict[test_pred_labels[i]] = 1
    wrong_dict = {k: v for k, v in sorted(wrong_dict.items(), key=lambda item: item[1], reverse=True)}
    print(wrong_dict)
    wrong_dict['Celltype'] = train_celltype
    wrong_dict['true_positive_rate'] = cell_pred/cell_count * 100

    return wrong_dict


def plot_embeddings_given_labels_and_datasets(train_embeddings ,test_embeddings, train_true_labels, test_true_labels, test_pred_labels, train_celltype, test_celltype, prototypes, prototypes_labels):
    cell_count = 0
    cell_pred = 0
    for i in range(len(test_pred_labels)):
        if test_true_labels[i] == test_celltype:
            cell_count += 1
            if test_pred_labels[i] == train_celltype:
                cell_pred += 1
    print("True positive " + test_celltype + ": " + str(cell_pred/cell_count * 100) + '%')

    wrong_dict = {}
    for i in range(len(test_true_labels)):
        if test_true_labels[i] == test_celltype and test_pred_labels[i] != train_celltype:
            if test_pred_labels[i] in wrong_dict.keys():
                wrong_dict[test_pred_labels[i]] += 1
            else:
                wrong_dict[test_pred_labels[i]] = 1
    wrong_dict = {k: v for k, v in sorted(wrong_dict.items(), key=lambda item: item[1], reverse=True)}
    print(wrong_dict)
    
    fig, ax = plt.subplots(figsize=(30, 20))
    i = np.where(np.array(test_true_labels) == test_celltype)
    ax.scatter(np.array(test_embeddings)[i,0], np.array(test_embeddings)[i,1], label="TEST DATASET")
    i = np.where(np.array(train_true_labels) == train_celltype)
    ax.scatter(np.array(train_embeddings)[i,0], np.array(train_embeddings)[i,1], label="TRAIN DATASET")
    for i in range(prototypes.shape[0]):
        ax.scatter(prototypes[i,0], prototypes[i,1], marker='x', s=100)
        ax.annotate(prototypes_labels[i], (prototypes[i,0], prototypes[i,1]))
    ax.legend()
    plt.show()


def plot_embeddings_scatter(embeddings, true_labels, embedding_prototypes, embedding_prototypes_labels):
    fig, ax = plt.subplots(figsize=(30, 20))
    for color in np.unique(np.array(true_labels)):
        i = np.where(np.array(true_labels) == color)
        ax.scatter(np.array(embeddings)[i,0], np.array(embeddings)[i,1], label=color)
    for i in range(embedding_prototypes.shape[0]):
        ax.scatter(embedding_prototypes[i,0], embedding_prototypes[i,1], marker='x', s=100)
        ax.annotate(embedding_prototypes_labels[i], (embedding_prototypes[i,0], embedding_prototypes[i,1]))
    ax.legend()
    plt.show()


def plot_embeddings_likelihood(model, encoder, pred_labels, embeddings, prototypes, prototypes_labels):
    targets = torch.index_select(model.prototypes, 0, torch.tensor(encoder.transform(pred_labels)))
    log_vars = torch.log(torch.index_select(model.vars, 0, torch.tensor(encoder.transform(pred_labels))))
    training_embedding_likelihoods = log_likelihood_student(torch.tensor(embeddings), targets, log_vars)
    training_embedding_likelihoods = np.array(training_embedding_likelihoods.detach())

    fig, ax = plt.subplots(figsize=(30, 20))
    plt.rcParams["figure.autolayout"] = True
    points = ax.scatter(np.array(embeddings)[:,0], np.array(embeddings)[:,1], c=training_embedding_likelihoods)
    fig.colorbar(points)
    for i in range(prototypes.shape[0]):
        ax.scatter(prototypes[i,0], prototypes[i,1], marker='x', s=100)
        ax.annotate(prototypes_labels[i], (prototypes[i,0], prototypes[i,1]))
    plt.show()