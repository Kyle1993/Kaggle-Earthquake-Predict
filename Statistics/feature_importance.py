import pandas as pd
import numpy as np
import random
import sys,os
import lightgbm as lgb
import pickle
from tqdm import tqdm

sys.path.append('../')
from global_variable import *


import warnings
warnings.filterwarnings("ignore")

with open('../kfold_{}.pkl'.format(num_samples),'rb') as f:
    kfold = pickle.load(f)

# train_data_df = pd.read_csv('train_data_(10000, 1805).csv',index_col=0)
train_data_df = pd.read_csv('/data2/jianglibin/earthquake/data/LGB/train_data_(10000, 2333).csv',index_col=0)

data = train_data_df.drop(columns=['ttf'],inplace=False)
label = train_data_df['ttf']

mix_feature_num = 1000
feature_range = 2333
top_n = 4
round_num = 5000
#
# mix_feature = []
#
# params = {
#     'boosting_type': 'gbdt',  # 训练方式
#     'objective': 'regression',
#     'metric': 'mae',  # 损失函数
#     'num_leaves':64,
#     'max_depth': -1,
#     'learning_rate': 0.01,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'verbosity': -1,
#     "bagging_freq": 1,
#     'lambda_l1': 0.2,
#     'lambda_l2': 0.2,
#     'seed':2019,
#     'n_estimators':300,
#     # 'device': 'gpu',
#     # 'gpu_platform_id': 0,
#     # 'gpu_device_id': 0
# }
#
# for i in tqdm(range(round_num)):
#     data_new_feature = pd.DataFrame()
#     col1 = list(data.columns)
#     col2 = list(data.columns)
#     random.shuffle(col1)
#     random.shuffle(col2)
#     col1 = col1[:mix_feature_num]
#     col2 = col2[:mix_feature_num]
#     for j in range(mix_feature_num):
#         type = random.randint(0, 3)
#         if type == 0:
#             col_name = '{} + {}'.format(col1[j], col2[j])
#             data_new_feature[col_name] = data[col1[j]] + data[col2[j]]
#         elif type == 1:
#             col_name = '{} - {}'.format(col1[j], col2[j])
#             data_new_feature[col_name] = data[col1[j]] - data[col2[j]]
#         elif type == 2:
#             col_name = '{} * {}'.format(col1[j], col2[j])
#             data_new_feature[col_name] = data[col1[j]] * data[col2[j]]
#         else:
#             col_name = '{} / {}'.format(col1[j], col2[j])
#             data_new_feature[col_name] = data[col1[j]] / data[col2[j]]
#
#     model = lgb.LGBMRegressor(**params,n_jobs=5)
#
#     model.fit(data_new_feature, label,eval_set=[(data_new_feature,label)], verbose=0)
#     res = model.evals_result_
#
#     feature_importance = pd.DataFrame()
#     feature_importance['feature'] = data_new_feature.columns
#     model.importance_type = 'gain'
#     feature_importance['importance'] = model.feature_importances_
#     feature_importance = feature_importance.sort_values(by='importance', ascending=False)
#
#     mix_feature.extend(feature_importance[:top_n]['feature'].values.tolist())


# with open('import_mix_feature_{}.pkl'.format(feature_range),'wb') as f:
#     pickle.dump(mix_feature,f)

with open('import_mix_feature_{}.pkl'.format(feature_range),'rb') as f:
    mix_feature = pickle.load(f)

mix_feature = list(set(mix_feature))


print('Add Feature:',len(mix_feature))
for mf in mix_feature:
    if '+' in mf:
        col1 = mf.split('+')[0].strip()
        col2 = mf.split('+')[1].strip()
        data[mf] = data[col1] + data[col2]
    elif '-' in mf:
        col1 = mf.split('-')[0].strip()
        col2 = mf.split('-')[1].strip()
        data[mf] = data[col1] - data[col2]
    elif '*' in mf:
        col1 = mf.split('*')[0].strip()
        col2 = mf.split('*')[1].strip()
        data[mf] = data[col1] * data[col2]
    elif '/' in mf:
        col1 = mf.split('/')[0].strip()
        col2 = mf.split('/')[1].strip()
        data[mf] = data[col1] / data[col2]

params = {
    'boosting_type': 'gbdt',  # 训练方式
    'objective': 'regression',
    'metric': 'mae',  # 损失函数
    'num_leaves':250,
    'max_depth': -1,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity': -1,
    # 'min_data_in_leaf': 100,
    "bagging_freq": 1,
    'lambda_l1': 0.2,
    'lambda_l2': 0.2,
    # 'min_sum_hessian_in_leaf':10,
    # 'max_bin':10,
    'seed':2019,
    'n_estimators':300,
    # 'device': 'gpu',
    # 'gpu_platform_id': 0,
    # 'gpu_device_id': 0
}

# early_stop_round = 100
fold = 0
train_idx = list(range(num_samples))
random.shuffle(train_idx)

if isinstance(data,np.ndarray):
    train_data = data[train_idx]
    train_label = label[train_idx]
else:
    train_data = data.iloc[train_idx]
    train_label = label.iloc[train_idx]

model = lgb.LGBMRegressor(**params,n_jobs=5)
model.fit(train_data,train_label,eval_set=[(train_data,train_label)],eval_metric='mae',verbose=10,early_stopping_rounds=50)
res = model.evals_result_


feature_importance = pd.DataFrame()
feature_importance['feature'] = train_data.columns
model.importance_type = 'split'
feature_importance['importance_split'] = model.feature_importances_
model.importance_type = 'gain'
feature_importance['importance_gain'] = model.feature_importances_
feature_importance = feature_importance[feature_importance['importance_split']>0]
feature_importance = feature_importance[feature_importance['importance_gain']>0]

feature_importance = feature_importance.sort_values(by='importance_gain',ascending=False)

print(feature_importance[:10])


feature_importance.to_csv('feature_importance_{}.csv'.format(feature_range))



