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


ray.init()
# dataset_popv = sc.read("/Volumes/SSD/global.h5ad")
dataset_popv = sc.read("/Volumes/SSD/popv_immune.h5ad")
dataset_popv = dataset_popv[dataset_popv.obs['cell_type'] != 'double-positive, alpha-beta thymocyte']
list_celltypes = dataset_popv.obs['cell_type'].unique().tolist()

# dataset_popv = sc.read("./pre_processed_datasets/celltypist_pca.h5ad")
# dataset_popv = sc.read("./pre_processed_datasets/popv_immune_pca.h5ad")
# list_celltypes = dataset_popv.obs['cell_type'].unique().tolist()

encoder_celltype = LabelEncoder()
encoder_celltype.fit(dataset_popv.obs['cell_type'])

list_ct = dataset_popv.obs['cell_type'].unique().tolist()
list_num_ct = encoder_celltype.transform(list_ct)
list_inner_nodes = ['popv_immune', 'myeloid leukocyte', 'mature B cell', 'NK', 'CD4', 'CD8']
all_nodes = list_ct + list_inner_nodes

encoder_celltype_inner = LabelEncoder()
encoder_celltype_inner.fit(list_inner_nodes)

# encoder_celltype_popv = LabelEncoder()
# encoder_celltype_popv.fit(dataset_popv.obs['cell_type'])

g = helper_fns.build_hierarchical_tree_popv_immune(all_nodes=all_nodes, list_ct=list_ct, list_inner_nodes=list_inner_nodes, encoder_celltype=encoder_celltype, encoder_celltype_inner=encoder_celltype_inner)

dist_df = helper_fns.get_dist_df(list_num_ct=list_num_ct, g=g)
D = torch.tensor(dist_df.values, dtype=float)

train_indices, test_indices, cv = helper_fns.costumized_train_test_split(dataset_popv, cross_validation=True, k_fold=5, obs_key='cell_type')
# cv = joblib.load("./results_0908/cv.pkl")
# Save the cv
joblib.dump(cv, "./results_popv/cv_popv.pkl")

# Define data loaders for training and testing data in this fold
encoders = {
    'obs': {
        'cell_type': encoder_celltype.transform
    }
}

@ray.remote
def get_results(i, cv, dataset_popv, encoders, D, list_num_ct):
    print("Fold: {}".format(i))

    print('PCA Start')
    pca = TruncatedSVD(n_components=128)
    pca.fit(dataset_popv[cv[i]['train']].X)
    dataset_pca = AnnData(pca.transform(dataset_popv.X))
    dataset_pca.obs = dataset_popv.obs
    dataset_popv = dataset_pca
    print('PCA Done')

    train_subsamplers = torch.utils.data.SubsetRandomSampler(cv[i]['train'])
    test_subsamplers = torch.utils.data.SubsetRandomSampler(cv[i]['test'])
    dataloader_trainings = AnnLoader(dataset_popv, batch_size=512, convert=encoders, sampler=train_subsamplers)
    dataloader_testings = AnnLoader(dataset_popv, batch_size=512, convert=encoders, sampler=test_subsamplers)

    results = pd.DataFrame(columns=['fold', 'model', 'accuracy', 'f1', 'recall', 'time', 'training_error', 'testing_error'])
    results_dict = {}

    weights = helper_fns.get_weights(num_celltypes=len(list_num_ct), dataset=dataset_popv, encoder=encoder_celltype, obs='cell_type')
    print('Proto_Net+disto_pl Start')
    t0 = time.time()
    model, error = train(mode='Proto_Net', loss_mode='disto_pl', epochs=50, embedding_dim=16, D=D, num_celltypes=list_num_ct, 
        encoder=encoder_celltype, dataset=dataset_popv, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings, 
        obs_name='cell_type', init_weights=weights)
    t1 = time.time()
    preds, embeddings = model(torch.tensor(dataset_popv[cv[i]['test']].X))
    preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    results_dict['proto_disto_pl_fold_{}'.format(i)] = preds
    results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Proto_Net+disto_pl', 
        'accuracy': accuracy_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds), 
        'recall': recall_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'f1': f1_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)
    print('Proto_Net+disto_pl Done')

    print('Proto_Net+disto Start')
    t0 = time.time()
    model, error = train(mode='Proto_Net', loss_mode='disto', epochs=50, embedding_dim=16, D=D, num_celltypes=list_num_ct, 
        encoder=encoder_celltype, dataset=dataset_popv, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings, 
        obs_name='cell_type', init_weights=weights)
    t1 = time.time()
    preds, embeddings = model(torch.tensor(dataset_popv[cv[i]['test']].X))
    preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    results_dict['proto_disto_fold_{}'.format(i)] = preds
    results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Proto_Net+disto', 
        'accuracy': accuracy_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds), 
        'recall': recall_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'f1': f1_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)
    print('Proto_Net+disto Done')

    print('Proto_Net+pl Start')
    t0 = time.time()
    model, error = train(mode='Proto_Net', loss_mode='pl', epochs=50, embedding_dim=16, D=D, num_celltypes=list_num_ct,
        encoder=encoder_celltype, dataset=dataset_popv, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings,
        obs_name='cell_type', init_weights=weights)
    t1 = time.time()
    preds, embeddings = model(torch.tensor(dataset_popv[cv[i]['test']].X))
    preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    results_dict['proto_pl_fold_{}'.format(i)] = preds
    results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Proto_Net+pl', 
        'accuracy': accuracy_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds), 
        'recall': recall_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'f1': f1_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)
    print('Proto_Net+pl Done')

    print('Proto_Net Start')
    t0 = time.time()
    model, error = train(mode='Proto_Net', loss_mode='', epochs=50, embedding_dim=16, D=D, num_celltypes=list_num_ct,
        encoder=encoder_celltype, dataset=dataset_popv, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings,
        obs_name='cell_type', init_weights=weights)
    t1 = time.time()
    preds, embeddings = model(torch.tensor(dataset_popv[cv[i]['test']].X))
    preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    results_dict['proto_fold_{}'.format(i)] = preds
    results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Proto_Net', 
        'accuracy': accuracy_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds), 
        'recall': recall_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'f1': f1_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)
    print('Proto_Net Done')

    print('Net Start')
    t0 = time.time()
    model, error = train(mode='Net', loss_mode='', epochs=30, embedding_dim=16, D=D, num_celltypes=list_num_ct, 
        encoder=encoder_celltype, dataset=dataset_popv, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings, 
        obs_name='cell_type', init_weights=weights)
    t1 = time.time()
    preds = model(torch.tensor(dataset_popv[cv[i]['test']].X))
    preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    results_dict['Net_fold_{}'.format(i)] = preds
    results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Net', 
        'accuracy': accuracy_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds), 
        'recall': recall_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'f1': f1_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)
    print('Net Done')

    print('logistic Start')
    t0 = time.time()
    model, error = train_logistic_regression(dataset=dataset_popv, train_indices=cv[i]['train'], test_indices=cv[i]['test'], encoder=encoder_celltype, obs_name='cell_type')
    t1 = time.time()
    preds = model.predict(dataset_popv[cv[i]['test']].X)
    preds = encoder_celltype.inverse_transform(preds)
    results_dict['logistic_fold_{}'.format(i)] = preds
    results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Logistic Regression',
        'accuracy': accuracy_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds), 
        'recall': recall_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'f1': f1_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)
    print('logistic Done')

    print('KNN Start')
    t0 = time.time()
    model = train_knn(dataset=dataset_popv, train_indices=cv[i]['train'], test_indices=cv[i]['test'], encoder=encoder_celltype, obs_name='cell_type')
    t1 = time.time()
    preds_train = model.predict(dataset_popv[cv[i]['train']].X)
    preds_train = encoder_celltype.inverse_transform(preds_train)
    preds = model.predict(dataset_popv[cv[i]['test']].X)
    preds = encoder_celltype.inverse_transform(preds)
    results_dict['knn_fold_{}'.format(i)] = preds
    results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'KNN',
        'accuracy': accuracy_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds), 
        'recall': recall_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'f1': f1_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds, average='weighted'), 
        'time': t1-t0, 'training_error':1 - accuracy_score(dataset_popv.obs['cell_type'][cv[i]['train']], preds_train), 
        'testing_error': 1 - accuracy_score(dataset_popv.obs['cell_type'][cv[i]['test']], preds)}, index=[0])], axis=0)
    print('KNN Done')

    return results, results_dict, i


# Start two tasks in the background.
cv0 = get_results.remote(0, cv, dataset_popv, encoders, D, list_num_ct)
cv1 = get_results.remote(1, cv, dataset_popv, encoders, D, list_num_ct)
cv2 = get_results.remote(2, cv, dataset_popv, encoders, D, list_num_ct)
cv3 = get_results.remote(3, cv, dataset_popv, encoders, D, list_num_ct)
cv4 = get_results.remote(4, cv, dataset_popv, encoders, D, list_num_ct)
# cv2 = cv.remote(2)
# cv3 = cv.remote(3)
# cv4 = cv.remote(4)

# Block until the tasks are done and get the results.
results_cv0, results_cv1, results_cv2, results_cv3, results_cv4 = ray.get([cv0, cv1, cv2, cv3, cv4])
# results_cv0, results_dict_cv0, i_cv0, results_cv1, results_dict_cv1, i_cv1 = ray.get([cv0, cv1])
# results_cv0,  = ray.get([cv0, cv1, cv2, cv3, cv4])

# results = pd.concat([results_cv0, results_cv1], axis=0)
# results_dict = {**results_dict_cv0, **results_dict_cv1}

# results.to_csv('results_celltypist.csv')
# with open('results_dict_celltypist.pickle', 'wb') as handle:
#     pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save results (tuple)
with open('./results_popv/results_popv.pickle', 'wb') as handle:
    pickle.dump((results_cv0, results_cv1, results_cv2, results_cv3, results_cv4), handle, protocol=pickle.HIGHEST_PROTOCOL)

