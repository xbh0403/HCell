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


ray.init()
dataset_celltypist = sc.read("/Users/xbh0403/Desktop/HCell/datasets/global.h5ad")
dataset_popv = sc.read("/Users/xbh0403/Desktop/HCell/datasets/popv_immune.h5ad")
list_celltypes = dataset_celltypist.obs['Manually_curated_celltype'].unique().tolist()
# list_celltypes = list(filter(lambda x: x not in [''], list_celltypes))
list_filtered_celltypes = list(filter(lambda x: 'doublets' not in x, list_celltypes))
dataset_celltypist = dataset_celltypist[dataset_celltypist.obs['Manually_curated_celltype'].isin(list_filtered_celltypes)]

train_overlap_celltypes = ['Memory B cells', 'Naive B cells', 'Tfh', 'Plasma cells', 'Plasmablasts', 'Tregs', 'Mast cells', 'Classical monocytes', 'Nonclassical monocytes']
test_overlap_celltypes = ['memory B cell', 'naive B cell', 'T follicular helper cell', 'plasma cell', 'plasmablast', 'regulatory T cell', 'mast cell', 'classical monocyte', 'non-classical monocyte']

# Keep only the test_overlap_celltypes for dataset_popv
dataset_popv = dataset_popv[dataset_popv.obs['cell_type'].isin(test_overlap_celltypes)]

for i in range(len(test_overlap_celltypes)):
    dataset_popv.obs['cell_type'] = dataset_popv.obs['cell_type'].replace(test_overlap_celltypes[i], train_overlap_celltypes[i])


# Filter the non-overlapping genes
ensembl_to_genesymbol = helper_fns.get_ensembl_mappings()
# save the ensembl_to_genesymbol
with open('./results_celltypist_415/ensembl_to_genesymbol.pkl', 'wb') as f:
    pickle.dump(ensembl_to_genesymbol, f)

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
encoder_celltype.fit(dataset_celltypist.obs['Manually_curated_celltype'])

list_ct = dataset_celltypist.obs['Manually_curated_celltype'].unique().tolist()
list_num_ct = encoder_celltype.transform(list_ct)
list_inner_nodes = ['Cross-tissue Immune Cell Atlas', 'B cell', 'Germinal center B cell', 'Myeloid', 'Dendritic cell',
                    'Macrophages', 'Monocytes', 'T & Innate lymphoid cells', 'CD4', 'T Naive', 'CD8', 
                    'Tissue-resident memory T (Trm) cells', 'NK']
all_nodes = list_ct + list_inner_nodes

encoder_celltype_inner = LabelEncoder()
encoder_celltype_inner.fit(list_inner_nodes)

encoder_celltype_popv = LabelEncoder()
encoder_celltype_popv.fit(dataset_popv.obs['cell_type'])

encoder_celltype_popv = LabelEncoder()
encoder_celltype_popv.fit(dataset_popv.obs['cell_type'])

if os.path.exists('./results_celltypist_415/g.pkl'):
    with open('./results_celltypist_415/g.pkl', 'rb') as f:
        g = pickle.load(f)
else:
    with open('./results_celltypist_415/g.pkl', 'wb') as f:
        g = helper_fns.build_hierarchical_tree_celltypist(all_nodes=all_nodes, list_ct=list_ct, list_inner_nodes=list_inner_nodes, encoder_celltype=encoder_celltype, encoder_celltype_inner=encoder_celltype_inner)
        pickle.dump(g, f)

if os.path.exists('./results_celltypist_415/g.png'):
    pass
else:
    layout = g.layout_reingold_tilford(mode='in', root=[helper_fns.transform('Cross-tissue Immune Cell Atlas', list_ct, list_inner_nodes, encoder_celltype, encoder_celltype_inner)])
    ig.plot(g, './results_celltypist_415/g.png', layout=layout, bbox=(0, 0, 5000, 500), vertex_label=g.vs['name'], vertex_label_size=10, vertex_label_dist=1.5, vertex_size=10, vertex_color='white', edge_color='grey', margin=50)
# dist_df = helper_fns.get_dist_df(list_num_ct=list_num_ct, g=g)
# D = torch.tensor(dist_df.values, dtype=float)

if os.path.exists('./results_celltypist_415/dist_df.csv'):
    dist_df = pd.read_csv('./results_celltypist_415/dist_df.csv', index_col=0)
else:
    dist_df = helper_fns.get_dist_df(list_num_ct=list_num_ct, g=g)
    dist_df.to_csv('./results_celltypist_415/dist_df.csv')
D = torch.tensor(dist_df.values, dtype=float)

# Chcek if the cv object is saved
if os.path.exists("./results_celltypist_415/cv.pkl"):
    cv = joblib.load("./results_celltypist_415/cv.pkl")
else:
    train_indices, test_indices, cv = helper_fns.costumized_train_test_split(dataset_celltypist, cross_validation=True, k_fold=5)
    joblib.dump(cv, "./results_celltypist_415/cv.pkl")

encoders = {
    'obs': {
        'Manually_curated_celltype': encoder_celltype.transform
    }
}

print("Normalize total & log1p")
sc.pp.normalize_total(dataset_celltypist, 1e4)
sc.pp.log1p(dataset_celltypist)
sc.pp.normalize_total(dataset_popv, 1e4)
sc.pp.log1p(dataset_popv)
print("Normalize total & log1p done")

@ray.remote
def get_results(i, cv, dataset_celltypist, dataset_popv, encoders, D, list_num_ct):
    print("Fold: {}".format(i))


    print('PCA Start')
    pca = TruncatedSVD(n_components=128)
    pca.fit(dataset_celltypist[cv[i]['train']].X)
    dataset_pca = AnnData(pca.transform(dataset_celltypist.X))
    dataset_pca.obs = dataset_celltypist.obs
    dataset_celltypist = dataset_pca

    dataset_popv_pca = AnnData(pca.transform(dataset_popv.X))
    dataset_popv_pca.obs = dataset_popv.obs
    dataset_popv = dataset_popv_pca

    # Check if /celltypist/models/PCA exists, if not, create it, then save the pca object
    if not os.path.exists("./results_celltypist_415/models/PCA"):
        os.makedirs("./results_celltypist_415/models/PCA")
    joblib.dump(pca, "./results_celltypist_415/models/PCA/pca_fold_{}.pkl".format(i))
    print('PCA Done')

    train_subsamplers = torch.utils.data.SubsetRandomSampler(cv[i]['train'])
    test_subsamplers = torch.utils.data.SubsetRandomSampler(cv[i]['test'])
    dataloader_trainings = AnnLoader(dataset_celltypist, batch_size=512, convert=encoders, sampler=train_subsamplers)
    dataloader_testings = AnnLoader(dataset_celltypist, batch_size=512, convert=encoders, sampler=test_subsamplers)

    results = pd.DataFrame(columns=['fold', 'model', 'accuracy', 'f1', 'recall', 'time', 'training_error', 'testing_error'])
    results_cross_dataset = pd.DataFrame(columns=['fold', 'model', 'accuracy', 'f1', 'recall'])
    results_dict = {}
    results_cross_dataset_dict = {}
    preds_and_score_overall = pd.DataFrame(columns=['model', 'fold', 'preds', 'score', 'true'])

    weights = helper_fns.get_weights(num_celltypes=len(list_num_ct), dataset=dataset_celltypist, encoder=encoder_celltype, obs='Manually_curated_celltype')

    print('Proto_Net+disto_pl Start')
    t0 = time.time()
    model, error = train(mode='Proto_Net', loss_mode='disto_pl', epochs=50, embedding_dim=16, D=D, num_celltypes=list_num_ct, 
        encoder=encoder_celltype, dataset=dataset_celltypist, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings, 
        obs_name='Manually_curated_celltype', init_weights=weights)
    t1 = time.time()
    # Check if /celltypist/models/Proto_Net_disto_pl exists, if not, create it, then save the model
    if not os.path.exists("./results_celltypist_415/models/Proto_Net_disto_pl"):
        os.makedirs("./results_celltypist_415/models/Proto_Net_disto_pl")
    torch.save(model.state_dict(), "./results_celltypist_415/models/Proto_Net_disto_pl/model_fold_{}.pt".format(i))
    # preds, embeddings = model(torch.tensor(dataset_celltypist[cv[i]['test']].X))
    # # Apply softmax to the predictions
    # preds = torch.nn.functional.softmax(preds, dim=1).detach().cpu().numpy()
    # preds_labels = encoder_celltype.inverse_transform(preds.argmax(axis=1))
    # preds_and_score = pd.DataFrame({'model': 'Proto_Net+disto_pl', 'fold': i, 'preds': preds_labels, 'score': preds.max(axis=1), 'true': dataset_celltypist[cv[i]['test']].obs['Manually_curated_celltype'].values})
    # preds_and_score_overall = pd.concat([preds_and_score_overall, preds_and_score])
    # preds_and_score['preds'] = preds_labels
    # preds_and_score['score'] = preds.argmax(axis=1)
    # preds_and_score['true'] = dataset_celltypist[cv[i]['test']].obs['Manually_curated_celltype'].values
    # results_dict['proto_disto_pl_fold_{}'.format(i)] = preds_and_score
    # results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Proto_Net+disto_pl', 
    #     'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels), 
    #     'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    #     'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    #     'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)

    preds, embeddings = model(torch.tensor(dataset_popv.X))
    preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    results_cross_dataset_dict['pred_proto_disto_pl_fold_{}'.format(i)] = preds
    results_cross_dataset_dict['true_proto_disto_pl_fold_{}'.format(i)] = dataset_popv.obs['cell_type'].values
    results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Proto_Net+disto_pl',
        'accuracy': accuracy_score(dataset_popv.obs['cell_type'], preds),
        'recall': recall_score(dataset_popv.obs['cell_type'], preds, average='weighted'),
        'f1': f1_score(dataset_popv.obs['cell_type'], preds, average='weighted')}, index=[0])], axis=0)

    print('Proto_Net+disto_pl Done')

    # print('Proto_Net+disto Start')
    # t0 = time.time()
    # model, error = train(mode='Proto_Net', loss_mode='disto', epochs=50, embedding_dim=16, D=D, num_celltypes=list_num_ct, 
    #     encoder=encoder_celltype, dataset=dataset_celltypist, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings, 
    #     obs_name='Manually_curated_celltype', init_weights=weights)
    # t1 = time.time()
    # # Check if /celltypist/models/Proto_Net_disto exists, if not, create it, then save the model
    # if not os.path.exists("./results_celltypist_415/models/Proto_Net_disto"):
    #     os.makedirs("./results_celltypist_415/models/Proto_Net_disto")
    # torch.save(model.state_dict(), "./results_celltypist_415/models/Proto_Net_disto/model_fold_{}.pt".format(i))
    # preds, embeddings = model(torch.tensor(dataset_celltypist[cv[i]['test']].X))
    # # Apply softmax to the predictions
    # preds = torch.nn.functional.softmax(preds, dim=1).detach().cpu().numpy()
    # preds_labels = encoder_celltype.inverse_transform(preds.argmax(axis=1))
    # preds_and_score = pd.DataFrame({'model': 'Proto_Net+disto', 'fold': i, 'preds': preds_labels, 'score': preds.max(axis=1), 'true': dataset_celltypist[cv[i]['test']].obs['Manually_curated_celltype'].values})
    # preds_and_score_overall = pd.concat([preds_and_score_overall, preds_and_score])
    # # preds_and_score['preds'] = preds_labels
    # # preds_and_score['score'] = preds.argmax(axis=1)
    # # results_dict['proto_disto_fold_{}'.format(i)] = preds_and_score
    # # results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Proto_Net+disto', 
    # #     'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels), 
    # #     'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    # #     'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    # #     'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)

    # preds, embeddings = model(torch.tensor(dataset_popv.X))
    # preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    # results_cross_dataset_dict['pred_proto_disto_fold_{}'.format(i)] = preds
    # results_cross_dataset_dict['true_proto_disto_fold_{}'.format(i)] = dataset_popv.obs['cell_type'].values
    # results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Proto_Net+disto',
    #     'accuracy': accuracy_score(dataset_popv.obs['cell_type'], preds),
    #     'recall': recall_score(dataset_popv.obs['cell_type'], preds, average='weighted'),
    #     'f1': f1_score(dataset_popv.obs['cell_type'], preds, average='weighted')}, index=[0])], axis=0)

    # print('Proto_Net+disto Done')

    # print('Proto_Net+pl Start')
    # t0 = time.time()
    # model, error = train(mode='Proto_Net', loss_mode='pl', epochs=50, embedding_dim=16, D=D, num_celltypes=list_num_ct,
    #     encoder=encoder_celltype, dataset=dataset_celltypist, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings,
    #     obs_name='Manually_curated_celltype', init_weights=weights)
    # t1 = time.time()
    # # Check if /celltypist/models/Proto_Net_pl exists, if not, create it, then save the model
    # if not os.path.exists("./results_celltypist_415/models/Proto_Net_pl"):
    #     os.makedirs("./results_celltypist_415/models/Proto_Net_pl")
    # torch.save(model.state_dict(), "./results_celltypist_415/models/Proto_Net_pl/model_fold_{}.pt".format(i))
    # preds, embeddings = model(torch.tensor(dataset_celltypist[cv[i]['test']].X))
    # # Apply softmax to the predictions
    # preds = torch.nn.functional.softmax(preds, dim=1).detach().cpu().numpy()
    # preds_labels = encoder_celltype.inverse_transform(preds.argmax(axis=1))
    # preds_and_score = pd.DataFrame({'model': 'Proto_Net+pl', 'fold': i, 'preds': preds_labels, 'score': preds.max(axis=1), 'true': dataset_celltypist[cv[i]['test']].obs['Manually_curated_celltype'].values})
    # preds_and_score_overall = pd.concat([preds_and_score_overall, preds_and_score])
    # preds_and_score['preds'] = preds_labels
    # preds_and_score['score'] = preds.argmax(axis=1)
    # results_dict['proto_pl_fold_{}'.format(i)] = preds_and_score
    # results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Proto_Net+pl', 
    #     'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels), 
    #     'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    #     'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    #     'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)

    # preds, embeddings = model(torch.tensor(dataset_popv.X))
    # preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    # results_cross_dataset_dict['pred_proto_pl_fold_{}'.format(i)] = preds
    # results_cross_dataset_dict['true_proto_pl_fold_{}'.format(i)] = dataset_popv.obs['cell_type'].values
    # results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Proto_Net+pl',
    #     'accuracy': accuracy_score(dataset_popv.obs['cell_type'], preds),
    #     'recall': recall_score(dataset_popv.obs['cell_type'], preds, average='weighted'),
    #     'f1': f1_score(dataset_popv.obs['cell_type'], preds, average='weighted')}, index=[0])], axis=0)
    
    # print('Proto_Net+pl Done')

    # print('Proto_Net Start')
    # t0 = time.time()
    # model, error = train(mode='Proto_Net', loss_mode='', epochs=50, embedding_dim=16, D=D, num_celltypes=list_num_ct,
    #     encoder=encoder_celltype, dataset=dataset_celltypist, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings,
    #     obs_name='Manually_curated_celltype', init_weights=weights)
    # t1 = time.time()
    # # Check if /celltypist/models/Proto_Net exists, if not, create it, then save the model
    # if not os.path.exists("./results_celltypist_415/models/Proto_Net"):
    #     os.makedirs("./results_celltypist_415/models/Proto_Net")
    # torch.save(model.state_dict(), "./results_celltypist_415/models/Proto_Net/model_fold_{}.pt".format(i))
    # preds, embeddings = model(torch.tensor(dataset_celltypist[cv[i]['test']].X))
    # Apply softmax to the predictions
    # preds = torch.nn.functional.softmax(preds, dim=1).detach().cpu().numpy()
    # preds_labels = encoder_celltype.inverse_transform(preds.argmax(axis=1))
    # preds_and_score = pd.DataFrame({'model': 'Proto_Net', 'fold': i, 'preds': preds_labels, 'score': preds.max(axis=1), 'true': dataset_celltypist[cv[i]['test']].obs['Manually_curated_celltype'].values})
    # preds_and_score_overall = pd.concat([preds_and_score_overall, preds_and_score])
    # preds_and_score['preds'] = preds_labels
    # preds_and_score['score'] = preds.argmax(axis=1)
    # results_dict['proto_fold_{}'.format(i)] = preds_and_score
    # results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Proto_Net', 
    #     'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels), 
    #     'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    #     'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    #     'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)

    # preds, embeddings = model(torch.tensor(dataset_popv.X))
    # preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    # results_cross_dataset_dict['pred_proto_fold_{}'.format(i)] = preds
    # results_cross_dataset_dict['true_proto_fold_{}'.format(i)] = dataset_popv.obs['cell_type'].values
    # results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Proto_Net',
    #     'accuracy': accuracy_score(dataset_popv.obs['cell_type'], preds),
    #     'recall': recall_score(dataset_popv.obs['cell_type'], preds, average='weighted'),
    #     'f1': f1_score(dataset_popv.obs['cell_type'], preds, average='weighted')}, index=[0])], axis=0)
    
    # print('Proto_Net Done')

    # print('Net Start')
    # t0 = time.time()
    # model, error = train(mode='Net_softmax', loss_mode='', epochs=30, embedding_dim=16, D=D, num_celltypes=list_num_ct, 
    #     encoder=encoder_celltype, dataset=dataset_celltypist, dataloader_training=dataloader_trainings, dataloader_testing=dataloader_testings, 
    #     obs_name='Manually_curated_celltype', init_weights=weights)
    # t1 = time.time()
    # # Check if /celltypist/models/Net exists, if not, create it, then save the model
    # if not os.path.exists("./results_celltypist_415/models/Net"):
    #     os.makedirs("./results_celltypist_415/models/Net")
    # torch.save(model.state_dict(), "./results_celltypist_415/models/Net/model_fold_{}.pt".format(i))
    # preds = model(torch.tensor(dataset_celltypist[cv[i]['test']].X))
    # preds = torch.nn.functional.softmax(preds, dim=1).detach().cpu().numpy()
    # preds_labels = encoder_celltype.inverse_transform(preds.argmax(axis=1))
    # preds_and_score = pd.DataFrame({'model': 'Net', 'fold': i, 'preds': preds_labels, 'score': preds.max(axis=1), 'true': dataset_celltypist[cv[i]['test']].obs['Manually_curated_celltype'].values})
    # preds_and_score_overall = pd.concat([preds_and_score_overall, preds_and_score])
    # preds_and_score['preds'] = preds_labels
    # preds_and_score['score'] = preds.argmax(axis=1)
    # results_dict['Net_fold_{}'.format(i)] = preds_and_score
    # results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Net', 
    #     'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels), 
    #     'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    #     'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    #     'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)
    
    # preds = model(torch.tensor(dataset_popv.X))
    # preds = encoder_celltype.inverse_transform(preds.detach().cpu().numpy().argmax(axis=1))
    # results_cross_dataset_dict['pred_Net_fold_{}'.format(i)] = preds
    # results_cross_dataset_dict['true_Net_fold_{}'.format(i)] = dataset_popv.obs['cell_type'].values
    # results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Net',
    #     'accuracy': accuracy_score(dataset_popv.obs['cell_type'], preds),
    #     'recall': recall_score(dataset_popv.obs['cell_type'], preds, average='weighted'),
    #     'f1': f1_score(dataset_popv.obs['cell_type'], preds, average='weighted')}, index=[0])], axis=0)

    # print('Net Done')

    # print('logistic Start')
    # t0 = time.time()
    # model, error = train_logistic_regression(dataset=dataset_celltypist, train_indices=cv[i]['train'], test_indices=cv[i]['test'], encoder=encoder_celltype, obs_name='Manually_curated_celltype')
    # t1 = time.time()
    # # Check if /celltypist/models/logistic_regression exists, if not, create it, then save the model
    # if not os.path.exists("./results_celltypist_415/models/logistic_regression"):
    #     os.makedirs("./results_celltypist_415/models/logistic_regression")
    # joblib.dump(model, "./results_celltypist_415/models/logistic_regression/model_fold_{}.pkl".format(i))
    # preds = model.predict_proba(dataset_celltypist[cv[i]['test']].X)
    # preds_labels = encoder_celltype.inverse_transform(preds.argmax(axis=1))
    # preds_and_score = pd.DataFrame({'model': 'logistic_regression', 'fold': i, 'preds': preds_labels, 'score': preds.max(axis=1), 'true': dataset_celltypist[cv[i]['test']].obs['Manually_curated_celltype'].values})
    # preds_and_score_overall = pd.concat([preds_and_score_overall, preds_and_score])
    # # preds_and_score['preds'] = preds_labels
    # # preds_and_score['score'] = preds.argmax(axis=1)
    # # results_dict['logistic_fold_{}'.format(i)] = preds_and_score
    # # results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'Logistic Regression',
    # #     'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels), 
    # #     'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    # #     'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    # #     'time': t1-t0, 'training_error': error['train'], 'testing_error': error['test']}, index=[0])], axis=0)
    
    # preds = model.predict(dataset_popv.X)
    # preds = encoder_celltype.inverse_transform(preds)
    # results_cross_dataset_dict['pred_logistic_fold_{}'.format(i)] = preds
    # results_cross_dataset_dict['true_logistic_fold_{}'.format(i)] = dataset_popv.obs['cell_type'].values
    # results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'Logistic Regression',
    #     'accuracy': accuracy_score(dataset_popv.obs['cell_type'], preds),
    #     'recall': recall_score(dataset_popv.obs['cell_type'], preds, average='weighted'),
    #     'f1': f1_score(dataset_popv.obs['cell_type'], preds, average='weighted')}, index=[0])], axis=0)
    
    # print('logistic Done')

    # print('KNN Start')
    # t0 = time.time()
    # model = train_knn(dataset=dataset_celltypist, train_indices=cv[i]['train'], test_indices=cv[i]['test'], encoder=encoder_celltype, obs_name='Manually_curated_celltype')
    # t1 = time.time()
    # # Check if /celltypist/models/KNN exists, if not, create it, then save the model
    # if not os.path.exists("./results_celltypist_415/models/KNN"):
    #     os.makedirs("./results_celltypist_415/models/KNN")
    # joblib.dump(model, "./results_celltypist_415/models/KNN/model_fold_{}.pkl".format(i))
    # # preds_train = model.predict(dataset_celltypist[cv[i]['train']].X)
    # # preds_train = encoder_celltype.inverse_transform(preds_train)
    # preds = model.predict_proba(dataset_celltypist[cv[i]['test']].X)
    # preds_labels = encoder_celltype.inverse_transform(preds.argmax(axis=1))
    # preds_and_score = pd.DataFrame({'model': 'KNN', 'fold': i, 'preds': preds_labels, 'score': preds.max(axis=1), 'true': dataset_celltypist[cv[i]['test']].obs['Manually_curated_celltype'].values})
    # preds_and_score_overall = pd.concat([preds_and_score_overall, preds_and_score])
    # # preds_and_score['preds'] = preds_labels
    # # preds_and_score['score'] = preds.argmax(axis=1)
    # # results_dict['KNN_fold_{}'.format(i)] = preds_and_score
    # # results = pd.concat([results, pd.DataFrame({'fold': i, 'model': 'KNN',
    # #     'accuracy': accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels), 
    # #     'recall': recall_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    # #     'f1': f1_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels, average='weighted'), 
    # #     'time': t1-t0, 'training_error':1 - accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['train']], preds_train), 
    # #     'testing_error': 1 - accuracy_score(dataset_celltypist.obs['Manually_curated_celltype'][cv[i]['test']], preds_labels)}, index=[0])], axis=0)
    
    # preds = model.predict(dataset_popv.X)
    # preds = encoder_celltype.inverse_transform(preds)
    # results_cross_dataset_dict['pred_knn_fold_{}'.format(i)] = preds
    # results_cross_dataset_dict['true_knn_fold_{}'.format(i)] = dataset_popv.obs['cell_type'].values
    # results_cross_dataset = pd.concat([results_cross_dataset, pd.DataFrame({'fold': i, 'model': 'KNN',
    #     'accuracy': accuracy_score(dataset_popv.obs['cell_type'], preds),
    #     'recall': recall_score(dataset_popv.obs['cell_type'], preds, average='weighted'),
    #     'f1': f1_score(dataset_popv.obs['cell_type'], preds, average='weighted')}, index=[0])], axis=0)
    
    # print('KNN Done')

    # return results, results_dict, i, results_cross_dataset, results_cross_dataset_dict
    # return preds_and_score_overall
    return results_cross_dataset, results_cross_dataset_dict


# Start two tasks in the background.
# i, cv, dataset_celltypist, dataset_popv, encoders, encoder_celltype, D, list_num_ct
cv0 = get_results.remote(0, cv, dataset_celltypist, dataset_popv, encoders, D, list_num_ct)
cv1 = get_results.remote(1, cv, dataset_celltypist, dataset_popv, encoders, D, list_num_ct)
cv2 = get_results.remote(2, cv, dataset_celltypist, dataset_popv, encoders, D, list_num_ct)
cv3 = get_results.remote(3, cv, dataset_celltypist, dataset_popv, encoders, D, list_num_ct)
cv4 = get_results.remote(4, cv, dataset_celltypist, dataset_popv, encoders, D, list_num_ct)
# cv2 = cv.remote(2)
# cv3 = cv.remote(3)
# cv4 = cv.remote(4)

# Block until the tasks are done and get the results.
results_cv0, results_cv1, results_cv2, results_cv3, results_cv4 = ray.get([cv0, cv1, cv2, cv3, cv4])
# results_cv0, results_dict_cv0, i_cv0, results_cv1, results_dict_cv1, i_cv1 = ray.get([cv0, cv1])
# results_cv0,  = ray.get([cv0, cv1, cv2, cv3, cv4])

# results = pd.concat([results_cv0, results_cv1], axis=0)
# results_dict = {**results_dict_cv0, **results_dict_cv1}

# results.to_csv('results_celltypist_415.csv')
# with open('results_dict_celltypist.pickle', 'wb') as handle:
#     pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)]
# preds_and_score_overall = pd.concat([results_cv0, results_cv1, results_cv2, results_cv3, results_cv4], axis=0)
# preds_and_score_overall.to_csv('preds_and_score_overall_celltypist.csv')

# Save results (tuple)
with open('./results_celltypist_415/results_celltypist_415.pickle', 'wb') as handle:
    pickle.dump((results_cv0, results_cv1, results_cv2, results_cv3, results_cv4), handle, protocol=pickle.HIGHEST_PROTOCOL)
