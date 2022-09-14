import helper_fns
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
import joblib
import ray

dataset_celltypist = sc.read("/Volumes/SSD/global.h5ad")
list_celltypes = dataset_celltypist.obs['Manually_curated_celltype'].unique().tolist()
list_celltypes = list(filter(lambda x: x not in ['Mast cells', 'pDC','Progenitor', 'Erythroid', 'Megakaryocytes'], list_celltypes))
list_filtered_celltypes = list(filter(lambda x: 'doublets' not in x, list_celltypes)) 
dataset_celltypist = dataset_celltypist[dataset_celltypist.obs['Manually_curated_celltype'].isin(list_filtered_celltypes)]

# dataset_celltypist = sc.read("./pre_processed_datasets/celltypist_pca.h5ad")
# dataset_popv = sc.read("./pre_processed_datasets/popv_immune_pca.h5ad")
# list_celltypes = dataset_celltypist.obs['Manually_curated_celltype'].unique().tolist()

encoder_celltype = LabelEncoder()
encoder_celltype.fit(dataset_celltypist.obs['Manually_curated_celltype'])

list_ct = dataset_celltypist.obs['Manually_curated_celltype'].unique().tolist()
list_num_ct = encoder_celltype.transform(list_ct)
list_inner_nodes = ['Cross-tissue Immune Cell Atlas', 'B cell', 'Germinal center B cell', 'Myeloid', 'Dendritic cell',
                    'Macrophages', 'Monocytes', 'T & Innate lymphoid cells', 'CD4', 'T Naive', 'CD8', 
                    'Tissue-resident memory T (Trm) cells', 'NK']
all_nodes = list_ct + list_inner_nodes

encoder_celltype_inner = LabelEncoder()
encoder_celltype_inner.fit(list_inner_nodes)

# encoder_celltype_popv = LabelEncoder()
# encoder_celltype_popv.fit(dataset_popv.obs['cell_type'])

graph_celltypist = helper_fns.build_hierarchical_tree_celltypist(all_nodes=all_nodes, list_ct=list_ct, list_inner_nodes=list_inner_nodes, encoder_celltype=encoder_celltype, encoder_celltype_inner=encoder_celltype_inner)
distance_matrix_celltypist = helper_fns.get_dist_df(list_num_ct=list_num_ct, g=graph_celltypist)
celltypes = []
for i in range(len(list_ct)):
    celltype = helper_fns.inverse_transform(distance_matrix_celltypist.index[i], list_ct=list_ct, encoder_celltype=encoder_celltype, encoder_celltype_inner=encoder_celltype_inner)
    celltypes.append(celltype)
distance_matrix_celltypist.index = celltypes
distance_matrix_celltypist.columns = celltypes

print('Trm_Th1/Th17' in distance_matrix_celltypist.index)
print('Trm_Th1/Th17' in distance_matrix_celltypist.columns)
print(distance_matrix_celltypist.loc['Trm_Th1/Th17', 'Trm_Th1/Th17'])

# load results_dict from results_0908
results_dict = joblib.load('./results_celltypist_cv/results_celltypist_dict.pickle')

celltypes_celltypist = dataset_celltypist.obs['Manually_curated_celltype'].values

all_models = ['KNN', 'Logistic Regression', 'Net', 'Proto_Net', 'Proto_Net+pl', 'Proto_Net+disto', 'Proto_Net+disto_pl']

# load cv
cv = joblib.load('results_celltypist_cv/cv.pkl')

all_keys = results_dict.keys()
keys_fold_0 = [key for key in all_keys if 'fold_0' in key]
keys_fold_1 = [key for key in all_keys if 'fold_1' in key]
keys_fold_2 = [key for key in all_keys if 'fold_2' in key]
keys_fold_3 = [key for key in all_keys if 'fold_3' in key]
keys_fold_4 = [key for key in all_keys if 'fold_4' in key]

true_labels_fold_0 = celltypes_celltypist[cv[0]['test']]
true_labels_fold_1 = celltypes_celltypist[cv[1]['test']]
true_labels_fold_2 = celltypes_celltypist[cv[2]['test']]
true_labels_fold_3 = celltypes_celltypist[cv[3]['test']]
true_labels_fold_4 = celltypes_celltypist[cv[4]['test']]

ray.init()

@ray.remote
def get_distance_matrix_celltypist(true_labels, keys_fold, fold_str, fold_num, distance_matrix_celltypist):
    df_dist = pd.DataFrame(columns=['model', 'fold', 'distance'])
    print(fold_str)
    total_num = len(true_labels) * len(keys_fold)
    print('Total num: {}'.format(total_num))
    for key in keys_fold:
        print(key)
        pred_labels = results_dict[key]
        for i in range(len(pred_labels)):
            dist = distance_matrix_celltypist.loc[true_labels[i], pred_labels[i]]
            df_dist = pd.concat([df_dist, pd.DataFrame([[key.split('_'+fold_str)[0], fold_num , dist]], columns=['model', 'fold', 'distance'])], ignore_index=True)
    return df_dist


cv0 = get_distance_matrix_celltypist.remote(true_labels_fold_0, keys_fold_0, 'fold_0', 0, distance_matrix_celltypist)
cv1 = get_distance_matrix_celltypist.remote(true_labels_fold_1, keys_fold_1, 'fold_1', 1, distance_matrix_celltypist)
cv2 = get_distance_matrix_celltypist.remote(true_labels_fold_2, keys_fold_2, 'fold_2', 2, distance_matrix_celltypist)
cv3 = get_distance_matrix_celltypist.remote(true_labels_fold_3, keys_fold_3, 'fold_3', 3, distance_matrix_celltypist)
cv4 = get_distance_matrix_celltypist.remote(true_labels_fold_4, keys_fold_4, 'fold_4', 4, distance_matrix_celltypist)

results_cv0, results_cv1, results_cv2, results_cv3, results_cv4 = ray.get([cv0, cv1, cv2, cv3, cv4])

results_cv = pd.concat([results_cv0, results_cv1, results_cv2, results_cv3, results_cv4])

results_cv.to_csv('./results_celltypist_cv/dist_df_celltypist.csv')