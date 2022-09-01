import numpy as np
import pandas as pd
import scanpy as sc
import torch
import math
import igraph as ig
import biomart

# Train test split & Cross-validation
def costumized_train_test_split(dataset, cross_validation=False, obs_key='Manually_curated_celltype', k_fold=5):
    indices_by_celltypes = {}
    train_indices, test_indices, cv = [], [], []
    for cell_type in dataset.obs[obs_key].unique():
        indices = np.where(dataset.obs[obs_key] == cell_type)[0]
        np.random.shuffle(indices)
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
def select_features(dataset_training, num_genes=36601):
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

                (transform('thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner), transform('double-positive, alpha-beta thymocyte', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)),
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
    server = biomart.BiomartServer('http://uswest.ensembl.org/biomart')
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

