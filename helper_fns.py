import numpy as np
import scanpy as sc

# Train test split & Cross-validation
def costumized_train_test_split(dataset, cross_validation=False, k_fold=5):
    indices_by_celltypes = {}
    train_indices, test_indices, cv = [], [], []
    for cell_type in dataset.obs['Manually_curated_celltype'].unique():
        indices = np.where(dataset.obs['Manually_curated_celltype'] == cell_type)[0]
        np.random.shuffle(indices)
        indices_by_celltypes.update({cell_type: indices})
        split = int(len(indices)/k_fold)
        if cross_validation:
            for i in range(k_fold):
                temp = i*split
                temp_test = list(indices[temp:temp+split])
                temp_train = list(set(indices) - set(temp_test))
                if cell_type != dataset.obs['Manually_curated_celltype'].unique()[0]:
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
    dataset_training.var['mt'] = dataset_training.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(dataset_training, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc_pp_train = sc.pp.filter_cells(dataset_training, min_genes=200, copy=True)
    sc.pp.filter_genes(sc_pp_train, min_cells=3)
    sc_pp_train = sc_pp_train[sc_pp_train.obs.n_genes_by_counts < 2500, :]
    sc_pp_train = sc_pp_train[sc_pp_train.obs.pct_counts_mt < 5, :]
    sc.pp.highly_variable_genes(sc_pp_train, n_top_genes=int(num_genes/4))
    sc_pp_train = sc_pp_train[:, sc_pp_train.var.highly_variable]
    return sc_pp_train