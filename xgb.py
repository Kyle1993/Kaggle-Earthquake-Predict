import pandas as pd
import numpy as np
import random
import sys,os
from functools import reduce
import time
import pickle
import utils
from xgboost import XGBRegressor

from global_variable import *

import warnings
warnings.filterwarnings("ignore")

with open('kfold_{}.pkl'.format(num_samples),'rb') as f:
    kfold = pickle.load(f)
assert len(set(kfold[0][0].tolist()+kfold[0][1].tolist())) == num_samples

with open('test_file.pkl','rb') as f:
    test_file = pickle.load(f)
test_file = [tf.split('.')[0] for tf in test_file]

handcraft_feature_num = 100
corr_check = True
ks_check = False

train_data, train_label, test_data, seg_id, data_mean, data_std, label_mean, label_std, feature_num = utils.load_merge_feature(handcraft_feature_num=handcraft_feature_num,corr_check=corr_check,drop_nainf=False,ks_check=ks_check)
assert seg_id.values.tolist() == test_file

params = {
    'booster':'gbtree',
    'objective':'reg:linear',
    'eval_metric':'mae',
    'gamma':0.01,
    # 'min_child_weight':1.1,
    'max_depth':7,
    'subsample':0.8,
    'colsample_bytree':0.8,
    # 'tree_method':'exact',
    'learning_rate':0.01,
    'n_estimators':300,
    'reg_alpha':0.3,
    'reg_lambda':0.3,
    # 'nthread':4,
    # 'scale_pos_weight':1,
    'seed':2019,
}


preds = []
record = []
pred_on_train = []
early_stop_round = 50
for fold,(train_idx,validate_idx) in enumerate(kfold):
    print('Fold{}:'.format(fold))
    random.shuffle(train_idx)
    random.shuffle(validate_idx)

    train_data_fold = train_data[fold][train_idx]
    train_label_fold = train_label[train_idx]
    validate_data_fold = train_data[fold][validate_idx]
    validate_label_fold = train_label[validate_idx]

    test_data_fold = test_data[fold]

    # train
    model = XGBRegressor(**params)
    model.fit(train_data_fold, train_label_fold, eval_set=[(train_data_fold, train_label_fold), (validate_data_fold, validate_label_fold)],eval_metric='mae', verbose=True,early_stopping_rounds=early_stop_round)
    res = model.evals_result()

    stop_round = np.asarray(res['validation_1']['mae']).argmin()

    # stop_round = len(res['validation_0']['mae']) - 1
    train_error = res['validation_0']['mae'][stop_round]
    validate_error = res['validation_1']['mae'][stop_round]
    record.append((stop_round,train_error,validate_error))

    print('stop round:{}\nstop train error:{}\nstop validate error:{}'.format(stop_round, train_error, validate_error))

    # test
    test_pred = model.predict(test_data_fold)
    test_pred = np.clip(test_pred, 0, 50)

    assert test_pred.shape[0] == test_num

    preds_fold = pd.DataFrame({'seg_id': seg_id, 'time_to_failure_{}'.format(fold): test_pred})
    preds_fold.set_index('seg_id', drop=True, inplace=True)
    preds.append(preds_fold)

    # pred on train
    pred_on_train_fold = model.predict(validate_data_fold)
    pred_on_train_fold = np.clip(pred_on_train_fold,0,50)
    pred_on_train_fold = pd.DataFrame(pred_on_train_fold)
    pred_on_train_fold['idx'] = validate_idx
    pred_on_train.append(pred_on_train_fold)

timestr = time.strftime('%m%d-%H%M')
# makedir
if not os.path.exists('XGB_{}'.format(timestr)):
    os.mkdir('XGB_{}'.format(timestr))

# record & parameter to log
error_mean = 0
log = 'Time: {}\n\n'.format(timestr)
log += 'Sample Num:{}\n'.format(num_samples)
for k,v in sorted(feature_num.items(),key=lambda x:x[0]):
    log += '{}:{}\n'.format(k,v)
log += '\nXGB Parameter:\n'
for k,v in sorted(params.items(),key=lambda x:x[0]):
    log += '{}: {}\n'.format(k,v)

log += '\n{:<5}\t{:<10}\t{:<20}\t{:<20}\n'.format('Fold','Stop Round','Train Error','Validate Error')
for i in range(cv):
    error_mean += record[i][2]
    log += '{:<5}\t{:<10}\t{:<20}\t{:<20}\n'.format(i,record[i][0],record[i][1],record[i][2])
error_mean /= cv
log += 'Mean CV Score: {}\n'.format(error_mean)

# generate submit
submit = reduce(lambda left, right: pd.merge(left, right, on='seg_id'),preds)
submit['time_to_failure'] = submit[['time_to_failure_{}'.format(i) for i in range(cv)]].mean(axis=1)
submit = submit[['time_to_failure']]
assert submit.shape[0] == test_num
submit.to_csv('XGB_{}/submit_{:.6f}.csv'.format(timestr,error_mean))

# utils.check_submit(submit)

pred_mean = submit['time_to_failure'].mean()
pred_std = submit['time_to_failure'].std()
log += '\nPred Mean:{}\nPred Std:{}\n'.format(pred_mean,pred_std)

with open('XGB_{}/log'.format(timestr),'w') as f:
    f.write(log)

# generate pred on train
pred_on_train = pd.concat(pred_on_train,ignore_index=True)
pred_on_train = pred_on_train.sort_values(by='idx')
assert pred_on_train['idx'].values.tolist() == list(range(num_samples))
pred_on_train.set_index('idx')
assert pred_on_train.shape == (num_samples,2)
pred_on_train.to_csv('XGB_{}/pred_on_train.csv'.format(timestr))


