import pandas as pd
import numpy as np
import time
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from anndata import AnnData
from anndata.experimental.pytorch import AnnLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import TruncatedSVD
import sklearn
import os
import joblib

import biomart
import prototypical_network
import helper_fns
# import vanilla_vae
from model import PL, Net, train, train_knn, train_logistic_regression
import ray
import igraph as ig
import model as m
import scipy as sp


ray.init()
dataset_celltypist = sc.read("/Volumes/SSD/global.h5ad")
dataset_popv = sc.read("/Volumes/SSD/popv_immune.h5ad")
list_celltypes = dataset_popv.obs['cell_type'].unique().tolist()
# These two are removed because the number of cells is too small
list_celltypes = list(filter(lambda x: x not in ['double-positive, alpha-beta thymocyte', 'myeloid dendritic cell'], list_celltypes))
dataset_popv = dataset_popv[dataset_popv.obs['cell_type'].isin(list_celltypes)]

test_overlap_celltypes = ['Memory B cells', 'Naive B cells', 'Tfh', 'Plasma cells', 'Plasmablasts', 'Tregs', 'Mast cells', 'Classical monocytes', 'Nonclassical monocytes']
train_overlap_celltypes = ['memory B cell', 'naive B cell', 'T follicular helper cell', 'plasma cell', 'plasmablast', 'regulatory T cell', 'mast cell', 'classical monocyte', 'non-classical monocyte']

# Keep only the test_overlap_celltypes for dataset_popv
dataset_celltypist = dataset_celltypist[dataset_celltypist.obs['Manually_curated_celltype'].isin(test_overlap_celltypes)]

for i in range(len(test_overlap_celltypes)):
    dataset_celltypist.obs['Manually_curated_celltype'] = dataset_celltypist.obs['Manually_curated_celltype'].replace(test_overlap_celltypes[i], train_overlap_celltypes[i])


# Filter the non-overlapping genes
ensembl_to_genesymbol = helper_fns.get_ensembl_mappings()

genes_celltypist = dataset_celltypist.var_names
genes_popv = dataset_popv.var_names
not_found = []
found_hgnc = []
found_ensembl = []
for i in range(len(list(genes_celltypist))):
    try:
        a = ensembl_to_genesymbol[genes_celltypist[i]]
        if a not in genes_popv:
            not_found.append(genes_celltypist[i])
        else:
            found_hgnc.append(genes_celltypist[i])
            found_ensembl.append(a)
    except KeyError:
        not_found.append(genes_celltypist[i])

# filter the anndata by the found_ensembl genes
dataset_popv = dataset_popv[:,found_ensembl]
dataset_celltypist = dataset_celltypist[:,found_hgnc]

encoder_celltype = LabelEncoder()
encoder_celltype.fit(dataset_popv.obs['cell_type'])

list_ct = dataset_popv.obs['cell_type'].unique().tolist()
list_num_ct = encoder_celltype.transform(list_ct)
list_inner_nodes = ['popv_immune', 'myeloid dendritic cell', 'myeloid leukocyte', 'mature B cell', 'NK', 'CD4', 'CD8']
all_nodes = list_ct + list_inner_nodes

encoder_celltype_inner = LabelEncoder()
encoder_celltype_inner.fit(list_inner_nodes)

encoder_celltype_celltypist = LabelEncoder()
encoder_celltype_celltypist.fit(dataset_celltypist.obs['Manually_curated_celltype'])

dist_df = pd.read_csv("./results_popv/dist_df.csv", index_col=0)

print("Normalize total & log1p")
sc.pp.normalize_total(dataset_celltypist, 1e4)
sc.pp.log1p(dataset_celltypist)
print("Normalize total & log1p done")

@ray.remote
def get_results(i, dataset_celltypist, dataset_popv):
    print("Fold: {}".format(i))


    print('PCA Start')
    # with open("./results_celltypist/models/PCA/pca_fold_{}.pkl".format(i)) as f:
    #     pca = joblib.load(f)
    pca = joblib.load("./results_popv/models/PCA/pca_fold_{}.pkl".format(i))
    
    dataset_celltypist_pca = AnnData(pca.transform(dataset_celltypist.X))
    dataset_celltypist_pca.obs = dataset_celltypist.obs
    dataset_celltypist = dataset_celltypist_pca


    results_cross_dataset = pd.DataFrame(columns=['fold', 'model', 'accuracy', 'f1', 'recall'])
    results_cross_dataset_dict = {}


    print('Proto_Net+disto_pl Start')
    # Load the model save by torch
    model = m.init_model(mode='Proto_Net', D=dist_df, embedding_dim=16, num_celltypes=list_num_ct)
    model.load_state_dict(torch.load("./results_popv/models/Proto_Net_disto_pl/model_fold_{}.pt".format(i)))

    preds, embeddings = model(torch.tensor(dataset_celltypist.X))
    preds_df = pd.DataFrame()
    for j in range(5):
        preds_df['pred_{}'.format(j)] = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argsort(axis=1)[:,-(j+1):-j if j != 0 else None])
    preds_df['true'] = dataset_celltypist.obs['Manually_curated_celltype'].tolist()
    preds_df['model'] = ['Proto_Net+disto_pl'] * len(preds_df['true'])
    match_index = []
    for k in range(preds_df.shape[0]):
        if preds_df['true'].tolist()[k] == preds_df['pred_0'].iloc[k]:
            match_index.append(0)
        elif preds_df['true'].tolist()[k] == preds_df['pred_1'].iloc[k]:
            match_index.append(1)
        elif preds_df['true'].tolist()[k] == preds_df['pred_2'].iloc[k]:
            match_index.append(2)
        elif preds_df['true'].tolist()[k] == preds_df['pred_3'].iloc[k]:
            match_index.append(3)
        elif preds_df['true'].tolist()[k] == preds_df['pred_4'].iloc[k]:
            match_index.append(4)
        else:
            match_index.append(5)
    preds_df['match_index'] = match_index
    # preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    results_cross_dataset_dict['proto_disto_pl_fold_{}'.format(i)] = preds_df
    results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Proto_Net+disto_pl',
        'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0']),
        'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted'),
        'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted')}, index=[0])], axis=0)

    print('Proto_Net+disto_pl Done')

    print('Proto_Net+disto Start')
    model = m.init_model(mode='Proto_Net', D=dist_df, embedding_dim=16, num_celltypes=list_num_ct)
    model.load_state_dict(torch.load("./results_popv/models/Proto_Net_disto/model_fold_{}.pt".format(i)))
    preds, embeddings = model(torch.tensor(dataset_celltypist.X))
    preds_df = pd.DataFrame()
    for j in range(5):
        preds_df['pred_{}'.format(j)] = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argsort(axis=1)[:,-(j+1):-j if j != 0 else None])
    # preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    preds_df['true'] = dataset_celltypist.obs['Manually_curated_celltype'].tolist()
    preds_df['model'] = ['Proto_Net+disto'] * len(preds_df['true'])
    match_index = []
    for k in range(preds_df.shape[0]):
        if preds_df['true'].tolist()[k] == preds_df['pred_0'].iloc[k]:
            match_index.append(0)
        elif preds_df['true'].tolist()[k] == preds_df['pred_1'].iloc[k]:
            match_index.append(1)
        elif preds_df['true'].tolist()[k] == preds_df['pred_2'].iloc[k]:
            match_index.append(2)
        elif preds_df['true'].tolist()[k] == preds_df['pred_3'].iloc[k]:
            match_index.append(3)
        elif preds_df['true'].tolist()[k] == preds_df['pred_4'].iloc[k]:
            match_index.append(4)
        else:
            match_index.append(5)
    preds_df['match_index'] = match_index
    results_cross_dataset_dict['proto_disto_fold_{}'.format(i)] = preds_df
    results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Proto_Net+disto',
        'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0']),
        'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted'),
        'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted')}, index=[0])], axis=0)

    print('Proto_Net+disto Done')

    print('Proto_Net+pl Start')
    model = m.init_model(mode='Proto_Net', D=dist_df, embedding_dim=16, num_celltypes=list_num_ct)
    model.load_state_dict(torch.load("./results_popv/models/Proto_Net_pl/model_fold_{}.pt".format(i)))

    preds, embeddings = model(torch.tensor(dataset_celltypist.X))
    # preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    preds_df = pd.DataFrame()
    for j in range(5):
        preds_df['pred_{}'.format(j)] = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argsort(axis=1)[:,-(j+1):-j if j != 0 else None])
    preds_df['true'] = dataset_celltypist.obs['Manually_curated_celltype'].tolist()
    preds_df['model'] = ['Proto_Net+pl'] * len(preds_df['true'])
    match_index = []
    for k in range(preds_df.shape[0]):
        if preds_df['true'].tolist()[k] == preds_df['pred_0'].iloc[k]:
            match_index.append(0)
        elif preds_df['true'].tolist()[k] == preds_df['pred_1'].iloc[k]:
            match_index.append(1)
        elif preds_df['true'].tolist()[k] == preds_df['pred_2'].iloc[k]:
            match_index.append(2)
        elif preds_df['true'].tolist()[k] == preds_df['pred_3'].iloc[k]:
            match_index.append(3)
        elif preds_df['true'].tolist()[k] == preds_df['pred_4'].iloc[k]:
            match_index.append(4)
        else:
            match_index.append(5)
    preds_df['match_index'] = match_index
    results_cross_dataset_dict['proto_pl_fold_{}'.format(i)] = preds_df
    results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Proto_Net+pl',
        'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0']),
        'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted'),
        'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted')}, index=[0])], axis=0)
    
    print('Proto_Net+pl Done')

    print('Proto_Net Start')
    model = m.init_model(mode='Proto_Net', D=dist_df, embedding_dim=16, num_celltypes=list_num_ct)
    model.load_state_dict(torch.load("./results_popv/models/Proto_Net/model_fold_{}.pt".format(i)))

    preds, embeddings = model(torch.tensor(dataset_celltypist.X))
    # preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    preds_df = pd.DataFrame()
    for j in range(5):
        preds_df['pred_{}'.format(j)] = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argsort(axis=1)[:,-(j+1):-j if j != 0 else None])
    preds_df['true'] = dataset_celltypist.obs['Manually_curated_celltype'].tolist()
    preds_df['model'] = ['Proto_Net'] * len(preds_df['true'])
    match_index = []
    for k in range(preds_df.shape[0]):
        if preds_df['true'].tolist()[k] == preds_df['pred_0'].iloc[k]:
            match_index.append(0)
        elif preds_df['true'].tolist()[k] == preds_df['pred_1'].iloc[k]:
            match_index.append(1)
        elif preds_df['true'].tolist()[k] == preds_df['pred_2'].iloc[k]:
            match_index.append(2)
        elif preds_df['true'].tolist()[k] == preds_df['pred_3'].iloc[k]:
            match_index.append(3)
        elif preds_df['true'].tolist()[k] == preds_df['pred_4'].iloc[k]:
            match_index.append(4)
        else:
            match_index.append(5)
    preds_df['match_index'] = match_index
    results_cross_dataset_dict['proto_fold_{}'.format(i)] = preds_df
    results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Proto_Net',
        'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0']),
        'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted'),
        'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted')}, index=[0])], axis=0)
    
    print('Proto_Net Done')

    print('Net Start')
    model = m.init_model(mode='Net', D=dist_df, embedding_dim=16, num_celltypes=list_num_ct)
    model.load_state_dict(torch.load("./results_popv/models/Net/model_fold_{}.pt".format(i)))
    
    preds = model(torch.tensor(dataset_celltypist.X))
    # preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    preds_df = pd.DataFrame()
    for j in range(5):
        preds_df['pred_{}'.format(j)] = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argsort(axis=1)[:,-(j+1):-j if j != 0 else None])
    preds_df['true'] = dataset_celltypist.obs['Manually_curated_celltype'].tolist()
    preds_df['model'] = ['Net'] * len(preds_df['true'])
    match_index = []
    for k in range(preds_df.shape[0]):
        if preds_df['true'].tolist()[k] == preds_df['pred_0'].iloc[k]:
            match_index.append(0)
        elif preds_df['true'].tolist()[k] == preds_df['pred_1'].iloc[k]:
            match_index.append(1)
        elif preds_df['true'].tolist()[k] == preds_df['pred_2'].iloc[k]:
            match_index.append(2)
        elif preds_df['true'].tolist()[k] == preds_df['pred_3'].iloc[k]:
            match_index.append(3)
        elif preds_df['true'].tolist()[k] == preds_df['pred_4'].iloc[k]:
            match_index.append(4)
        else:
            match_index.append(5)
    preds_df['match_index'] = match_index
    results_cross_dataset_dict['Net_fold_{}'.format(i)] = preds_df
    results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Net',
        'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0']),
        'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted'),
        'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted')}, index=[0])], axis=0)

    print('Net Done')

    print('logistic Start')
    # with open('./results_celltypist/models/logistic_regression/model_fold_{}.pkl'.format(i), 'rb') as f:
    #     model = pickle.load(f)
    model = joblib.load('./results_popv/models/logistic_regression/model_fold_{}.pkl'.format(i))
    
    preds = model.predict_proba(dataset_celltypist.X)
    # preds = encoder_celltype.inverse_transform(preds)
    preds_df = pd.DataFrame()
    for j in range(5):
        preds_df['pred_{}'.format(j)] = encoder_celltype.inverse_transform(preds.argsort(axis=1)[:,-(j+1):-j if j != 0 else None])
    preds_df['true'] = dataset_celltypist.obs['Manually_curated_celltype'].tolist()
    preds_df['model'] = ['logistic'] * len(preds_df['true'])
    match_index = []
    for k in range(preds_df.shape[0]):
        if preds_df['true'].tolist()[k] == preds_df['pred_0'].iloc[k]:
            match_index.append(0)
        elif preds_df['true'].tolist()[k] == preds_df['pred_1'].iloc[k]:
            match_index.append(1)
        elif preds_df['true'].tolist()[k] == preds_df['pred_2'].iloc[k]:
            match_index.append(2)
        elif preds_df['true'].tolist()[k] == preds_df['pred_3'].iloc[k]:
            match_index.append(3)
        elif preds_df['true'].tolist()[k] == preds_df['pred_4'].iloc[k]:
            match_index.append(4)
        else:
            match_index.append(5)
    preds_df['match_index'] = match_index
    results_cross_dataset_dict['logistic_fold_{}'.format(i)] = preds_df
    results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Logistic Regression',
        'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0']),
        'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted'),
        'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted')}, index=[0])], axis=0)
    
    print('logistic Done')

    print('KNN Start')
    # with open('./results_celltypist/models/KNN/model_fold_{}.pkl'.format(i), 'rb') as f:
    #     model = pickle.load(f)
    model = joblib.load('./results_popv/models/KNN/model_fold_{}.pkl'.format(i))

    preds = model.predict_proba(dataset_celltypist.X)
    # preds = encoder_celltype.inverse_transform(preds)
    preds_df = pd.DataFrame()
    for j in range(5):
        preds_df['pred_{}'.format(j)] = encoder_celltype.inverse_transform(preds.argsort(axis=1)[:,-(j+1):-j if j != 0 else None])
    preds_df['true'] = dataset_celltypist.obs['Manually_curated_celltype'].tolist()
    preds_df['model'] = ['KNN'] * len(preds_df['true'])
    match_index = []
    for k in range(preds_df.shape[0]):
        if preds_df['true'].tolist()[k] == preds_df['pred_0'].iloc[k]:
            match_index.append(0)
        elif preds_df['true'].tolist()[k] == preds_df['pred_1'].iloc[k]:
            match_index.append(1)
        elif preds_df['true'].tolist()[k] == preds_df['pred_2'].iloc[k]:
            match_index.append(2)
        elif preds_df['true'].tolist()[k] == preds_df['pred_3'].iloc[k]:
            match_index.append(3)
        elif preds_df['true'].tolist()[k] == preds_df['pred_4'].iloc[k]:
            match_index.append(4)
        else:
            match_index.append(5)
    preds_df['match_index'] = match_index
    results_cross_dataset_dict['knn_fold_{}'.format(i)] = preds_df
    results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'KNN',
        'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0']),
        'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted'),
        'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'], preds_df['pred_0'], average='weighted')}, index=[0])], axis=0)
    
    print('KNN Done')

    return results_cross_dataset, results_cross_dataset_dict


# Start two tasks in the background.
# i, cv, dataset_celltypist, dataset_popv, encoders, encoder_celltype, D, list_num_ct
cv0 = get_results.remote(0, dataset_celltypist, dataset_popv)
cv1 = get_results.remote(1, dataset_celltypist, dataset_popv)
cv2 = get_results.remote(2, dataset_celltypist, dataset_popv)
cv3 = get_results.remote(3, dataset_celltypist, dataset_popv)
cv4 = get_results.remote(4, dataset_celltypist, dataset_popv)
# cv2 = cv.remote(2)
# cv3 = cv.remote(3)
# cv4 = cv.remote(4)

# Block until the tasks are done and get the results.
results_cv0, results_cv1, results_cv2, results_cv3, results_cv4 = ray.get([cv0, cv1, cv2, cv3, cv4])

# Save results (tuple)
with open('./results_popv/results_preds_top5_true_model_crossdataset_popv.pickle', 'wb') as handle:
    pickle.dump((results_cv0, results_cv1, results_cv2, results_cv3, results_cv4), handle, protocol=pickle.HIGHEST_PROTOCOL)

