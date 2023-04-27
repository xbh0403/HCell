import joblib
import pandas as pd
import pickle

results = joblib.load('./results_celltypist/results_celltypist.pickle')
df_list = []
dict_list = []
for i in range(len(results)):
    df_list.append(results[i][0])
    dict_list.append(results[i][1])
results_df = pd.concat(df_list)
# Check if training_error and testing_error are smaller than 1, if so, multiply by 100
results_df['training_error'] = results_df['training_error'].apply(lambda x: x*100 if x < 1 else x)
results_df['testing_error'] = results_df['testing_error'].apply(lambda x: x*100 if x < 1 else x)
results_df.to_csv('./results_celltypist/results_celltypist.csv')


# Concat dict list into one dict
results_dict = {}
for d in dict_list:
    for k, v in d.items():
        results_dict[k] = v
with open('./results_celltypist/results_celltypist_dict.pickle', 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



# Cross-dataset
df_list = []
dict_list = []
for i in range(len(results)):
    df_list.append(results[i][3])
    dict_list.append(results[i][4])
results_df = pd.concat(df_list)
results_df.to_csv('./results_celltypist/results_celltypist_popv.csv')


# Concat dict list into one dict
results_dict = {}
for d in dict_list:
    for k, v in d.items():
        results_dict[k] = v
with open('./results_celltypist/results_celltypist_popv_dict.pickle', 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)