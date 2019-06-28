import pandas as pd
import numpy as np
import sys,os
# from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle
from functools import reduce
import time
from global_variable import *


with open('test_file.pkl','rb') as f:
    test_file = pickle.load(f)
test_file = [tf.split('.')[0] for tf in test_file]
with open('kfold_{}.pkl'.format(num_samples),'rb') as f:
    kfold = pickle.load(f)

# model_folder_list = ['LGB_0514-0905',
#                      'LGB_0513-1323',
#                      'NN_0513-1302',
#                      'RF_0513-1707',
#                      'SVR_0520-1516',
#                      'XGB_0513-1820',
#                      'XGB_0515-2136',
#                      'XGB_0516-2335',
#                      'XGB_0520-1009',
#                      'XGB_0520-1014',
#                      'XGB_0520-1357',
#                      'XGB_0520-1409',]

# model_folder_list = ['LGB_0521-1451',
#                      'NN_0521-1452',
#                      'RF_0521-1454',
#                      'SVR_0521-1455',
#                      'XGB_0521-1457',]

# model_folder_list = ['LGB_0522-0926',
#                      'NN_0522-0927',
#                      'RF_0522-0931',
#                      'SVR_0522-0933',
#                      'XGB_0522-0937',]

# model_folder_list = ['LGB_0523-1151',
#                      'NN_0523-1137',
#                      'RF_0523-1154',
#                      'SVR_0523-1157',
#                      'XGB_0523-1204',]

# model_folder_list = ['LGB_0531-2346',
#                      'NN_0531-2348',
#                      'RF_0531-2351',
#                      'SVR_0531-2355',
#                      'XGB_0531-2359',]

# hand = 20
model_folder_list = ['LGB_0601-2252',
                     'NN_0601-2257',
                     'RF_0601-2308',
                     'SVR_0601-2316',
                     'XGB_0601-2321',]

label_path = '/data2/jianglibin/earthquake/data/LGB/train_data_(10000, 1805).csv'
data = pd.read_csv(label_path)
label = pd.DataFrame(data['ttf'])
label['idx'] = np.arange(num_samples)

pred_on_train = [label]
pred_on_test = []
print('Merge Folder:')
for mfl in model_folder_list:
    files = os.listdir(mfl)
    submit_file = ''
    for f in files:
        if 'submit' in f:
            submit_file = f
    pred_on_test_ = pd.read_csv(os.path.join(mfl,submit_file),)
    pred_on_train_ = pd.read_csv(os.path.join(mfl,'pred_on_train.csv'),index_col=0)
    print(mfl,pred_on_train_.shape)
    assert pred_on_train_.shape == (num_samples, 2)
    assert pred_on_test_.shape == (test_num, 2)

    pred_on_test.append(pred_on_test_)
    pred_on_train.append(pred_on_train_)

# merge
pred_on_train = reduce(lambda left,right:pd.merge(left,right,on='idx'),pred_on_train)
pred_on_test = reduce(lambda left,right:pd.merge(left,right,on='seg_id'),pred_on_test)

assert pred_on_train.shape == (num_samples, len(model_folder_list)+2)
assert pred_on_test.shape == (test_num, len(model_folder_list)+1)
assert pred_on_train['idx'].tolist() == list(range(num_samples))
assert pred_on_test['seg_id'].tolist() == test_file

submit = pd.DataFrame()
submit['seg_id'] = pred_on_test['seg_id']

# train & inference
pred_on_train_data = pred_on_train.drop(columns=['idx','ttf'])
pred_on_train_label = pred_on_train['ttf']
preds = []
record = []
for fold in range(cv):
    train_idxs = kfold[fold][0]
    validate_idxs = kfold[fold][1]

    train_data = pred_on_train_data.iloc[train_idxs]
    train_label = pred_on_train_label.iloc[train_idxs]
    validate_data = pred_on_train_data.iloc[validate_idxs]
    validate_label = pred_on_train_label.iloc[validate_idxs]

    # model = BaggingRegressor(n_estimators=3,max_samples=0.5,max_features=0.5)
    model = LinearRegression()
    model.fit(train_data,train_label)

    train_pred = model.predict(train_data)
    train_error = mean_absolute_error(train_label,train_pred)

    validate_pred = model.predict(validate_data)
    validate_error = mean_absolute_error(validate_label,validate_pred)
    record.append([train_error,validate_error])
    print('Train Error:{}\tValidate Error:{}'.format(train_error,validate_error))


    test_pred = model.predict(pred_on_test.drop(columns=['seg_id']))
    preds.append(test_pred)

preds = np.asarray(preds).mean(axis=0)
submit['time_to_failure'] = preds
submit.set_index('seg_id', drop=True, inplace=True)
assert submit.shape == (test_num,1)

timestr = time.strftime('%m%d-%H%M')

record = np.asarray(record)
mean_error = record[:,1].mean()
print('Mean Error:',mean_error)
if not os.path.exists('ensemble_submit/submit_{}'.format(timestr)):
    os.mkdir('ensemble_submit/submit_{}'.format(timestr))

log = 'Time:{}\n\n'.format(timestr)
log += 'Ensemble Folder:\n'
for mlf in model_folder_list:
    log += '{}\n'.format(mlf)
log += '\n'
for i in range(cv):
    log += 'Train Error:{}\tValidate Error:{}\n'.format(record[i,0],record[i,1])
log += '\nMean Train Error:{}\nMean Validate Error:{}\n'.format(record[:,0].mean(),record[:,1].mean())
log += '\nNote:\n'

with open('ensemble_submit/submit_{}/log'.format(timestr),'w') as f:
    f.write(log)
submit.to_csv('ensemble_submit/submit_{}/submit_{:.6f}.csv'.format(timestr,mean_error))











