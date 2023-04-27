import helper_fns
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
import joblib
import ray
import os


dataset_popv = sc.read("/Volumes/SSD/popv_immune.h5ad")
list_celltypes = dataset_popv.obs['cell_type'].unique().tolist()
# These two are removed because the number of cells is too small
list_celltypes = list(filter(lambda x: x not in ['double-positive, alpha-beta thymocyte', 'myeloid dendritic cell'], list_celltypes))
dataset_popv = dataset_popv[dataset_popv.obs['cell_type'].isin(list_celltypes)]

# dataset_popv = sc.read("./pre_processed_datasets/celltypist_pca.h5ad")
# dataset_popv = sc.read("./pre_processed_datasets/popv_immune_pca.h5ad")
# list_celltypes = dataset_popv.obs['cell_type'].unique().tolist()

encoder_celltype = LabelEncoder()
encoder_celltype.fit(dataset_popv.obs['cell_type'])

list_ct = dataset_popv.obs['cell_type'].unique().tolist()
list_num_ct = encoder_celltype.transform(list_ct)
list_inner_nodes = ['popv_immune', 'myeloid dendritic cell', 'myeloid leukocyte', 'mature B cell', 'NK', 'CD4', 'CD8']
all_nodes = list_ct + list_inner_nodes

encoder_celltype_inner = LabelEncoder()
encoder_celltype_inner.fit(list_inner_nodes)

# encoder_celltype_popv = LabelEncoder()
# encoder_celltype_popv.fit(dataset_popv.obs['cell_type'])
distance_matrix_popv = pd.read_csv("./results_popv/dist_df.csv", index_col=0)
celltypes = []
for i in range(len(list_ct)):
    celltype = helper_fns.inverse_transform(distance_matrix_popv.index[i], list_ct=list_ct, encoder_celltype=encoder_celltype, encoder_celltype_inner=encoder_celltype_inner)
    celltypes.append(celltype)
distance_matrix_popv.index = celltypes
distance_matrix_popv.columns = celltypes

# load results_dict from results_0908
results_dict = joblib.load('./results_popv/results_popv_dict.pickle')

celltypes_popv = dataset_popv.obs['cell_type'].values

all_models = ['KNN', 'Logistic Regression', 'Net', 'Proto_Net', 'Proto_Net+pl', 'Proto_Net+disto', 'Proto_Net+disto_pl']

# load cv
cv = joblib.load('results_popv/cv.pkl')

all_keys = results_dict.keys()
keys_fold_0 = [key for key in all_keys if 'fold_0' in key]
keys_fold_1 = [key for key in all_keys if 'fold_1' in key]
keys_fold_2 = [key for key in all_keys if 'fold_2' in key]
keys_fold_3 = [key for key in all_keys if 'fold_3' in key]
keys_fold_4 = [key for key in all_keys if 'fold_4' in key]

true_labels_fold_0 = celltypes_popv[cv[0]['test']]
orig_indices_fold_0 = cv[0]['test']
true_labels_fold_1 = celltypes_popv[cv[1]['test']]
orig_indices_fold_1 = cv[1]['test']
true_labels_fold_2 = celltypes_popv[cv[2]['test']]
orig_indices_fold_2 = cv[2]['test']
true_labels_fold_3 = celltypes_popv[cv[3]['test']]
orig_indices_fold_3 = cv[3]['test']
true_labels_fold_4 = celltypes_popv[cv[4]['test']]
orig_indices_fold_4 = cv[4]['test']

ray.init()

@ray.remote
def get_distance_matrix(true_labels, orig_index, keys_fold, fold_str, fold_num, distance_matrix):
    df_dist = pd.DataFrame(columns=['model', 'fold', 'distance', 'true', 'pred', 'orig_index'])
    print(fold_str)
    total_num = len(true_labels) * len(keys_fold)
    print('Total num: {}'.format(total_num))
    for key in keys_fold:
        print(key)
        pred_labels = results_dict[key]
        for i in range(len(pred_labels)):
            dist = distance_matrix.loc[true_labels[i], pred_labels[i]]
            df_dist = pd.concat([df_dist, pd.DataFrame([[key.split('_'+fold_str)[0], fold_num , dist, true_labels[i], pred_labels[i], orig_index[i]]], columns=['model', 'fold', 'distance', 'true', 'pred', 'orig_index'])], ignore_index=True)
    return df_dist


cv0 = get_distance_matrix.remote(true_labels_fold_0, orig_indices_fold_0, keys_fold_0, 'fold_0', 0, distance_matrix_popv)
cv1 = get_distance_matrix.remote(true_labels_fold_1, orig_indices_fold_1, keys_fold_1, 'fold_1', 1, distance_matrix_popv)
cv2 = get_distance_matrix.remote(true_labels_fold_2, orig_indices_fold_2, keys_fold_2, 'fold_2', 2, distance_matrix_popv)
cv3 = get_distance_matrix.remote(true_labels_fold_3, orig_indices_fold_3, keys_fold_3, 'fold_3', 3, distance_matrix_popv)
cv4 = get_distance_matrix.remote(true_labels_fold_4, orig_indices_fold_4, keys_fold_4, 'fold_4', 4, distance_matrix_popv)

results_cv0, results_cv1, results_cv2, results_cv3, results_cv4 = ray.get([cv0, cv1, cv2, cv3, cv4])

results_cv = pd.concat([results_cv0, results_cv1, results_cv2, results_cv3, results_cv4])

results_cv.to_csv('./results_popv/results_popv_dists.csv')