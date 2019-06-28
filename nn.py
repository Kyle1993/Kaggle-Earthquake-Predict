import pandas as pd
import numpy as np
import sys,os
import time
import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
import pickle
from functools import reduce
import utils

from DL.models import Earthquake_NN
from DL.config import *
from DL.dataset_helper import EarthQuakeDataset,EarthQuakeDataset_test
from DL import dl_utils

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


timestr = time.strftime('%m%d-%H%M')
# makedir
if not os.path.exists('NN_{}'.format(timestr)):
    os.mkdir('NN_{}'.format(timestr))

config = Config()
config.lr = 1e-4
config.epoch = 20
config.feature_num = feature_num['Total num']

record = 'Time:{}\n\n'.format(timestr)
record += 'Sample Num:{}\n\n'.format(num_samples)
for k,v in sorted(feature_num.items(),key=lambda x:x[0]):
    record += '{}:{}\n'.format(k,v)
record += config.to_str()
record += '\n{:<5}\t{:<20}\t{:<20}\n'.format('Fold','Train Error','Validate Error')

validate_mean_error = 0
preds = []
pred_on_train = [] # use for model ensemble
for fold in range(cv):
    print('Fold {}: training {} epoch'.format(fold,config.epoch))
    train_idxs = kfold[fold][0]
    validate_idxs = kfold[fold][1]

    train_data_fold = train_data[fold]
    train_label_fold = train_label
    test_data_fold = test_data[fold]

    # normalize
    train_data_fold = (train_data_fold - data_mean) / data_std
    train_label_fold = (train_label_fold - label_mean) / label_std
    test_data_fold = (test_data_fold - data_mean) / data_std

    # init model & criticer & optimizer
    model = Earthquake_NN(feature_num['Total num'])
    model.cuda(config.gpu)
    criticer = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=config.lr)

    # start training
    model.train()
    train_y = []
    train_pred = []
    for epoch in tqdm(range(config.epoch)):
        trainset = EarthQuakeDataset(train_data_fold, train_label_fold, train_idxs)
        trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

        for _,traindata,trainlabel in trainloader:
            traindata = dl_utils.totensor(traindata).float()
            trainlabel = dl_utils.totensor(trainlabel).float()
            pred = model(traindata)
            if epoch == config.epoch -1:
                train_y.append(trainlabel.data.cpu().numpy())
                train_pred.append(pred.data.cpu().numpy())
            loss = criticer(pred,trainlabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_y = np.concatenate(train_y,axis=0)
    train_pred = np.concatenate(train_pred,axis=0)
    train_error = mean_absolute_error(train_y,train_pred) * label_std

    # validate
    model.eval()
    validate_y = []
    validate_pred = []
    validate_idx = []
    validateset = EarthQuakeDataset(train_data_fold, train_label_fold, validate_idxs)
    validateloader = DataLoader(validateset, batch_size=config.validate_batch_size, shuffle=False)
    with torch.no_grad():
        for idx,validatedata,validatelabel in tqdm(validateloader):
            validatedata = dl_utils.totensor(validatedata).float()
            pred = model(validatedata)
            validate_pred.append(pred.cpu().numpy())
            validate_y.append(validatelabel)
            validate_idx.extend(idx)
    validate_y = np.concatenate(validate_y,axis=0)
    validate_pred = np.concatenate(validate_pred,axis=0)
    validate_error = mean_absolute_error(validate_y,validate_pred) * label_std
    validate_pred = pd.DataFrame(validate_pred*label_std+label_mean)
    validate_pred['idx'] = np.asarray(validate_idx)
    pred_on_train.append(validate_pred)

    # pred test
    model.eval()
    test_dataset = EarthQuakeDataset_test(test_data_fold,seg_id)
    testloader = DataLoader(test_dataset,batch_size=128)
    testpreds = []
    testnames = []
    with torch.no_grad():
        for testdata, testname in tqdm(testloader):
            testdata = dl_utils.totensor(testdata).float()
            testpred = model(testdata)
            testpreds.append(testpred.cpu().numpy())
            testnames.extend(testname)

    testpreds = np.concatenate(testpreds,axis=0)
    testpreds = testpreds*label_std + label_mean
    testpreds = np.clip(testpreds,0,50)

    assert testpreds.shape == (test_num,)
    assert test_file == testnames

    preds_fold = pd.DataFrame({'seg_id':testnames,'time_to_failure_{}'.format(fold):testpreds})
    preds_fold.set_index('seg_id',drop=True,inplace=True)
    preds.append(preds_fold)

    print('Fold {}:\tTrain Error:{:.6f}\tValidate Error:{:.6f}'.format(fold,train_error,validate_error))
    record += '{:<5}\t{:<20}\t{:<20}\n'.format(fold,train_error,validate_error)
    validate_mean_error += validate_error

    # # save fold model
    # torch.save(model.cpu().state_dict(),'NN_{}/fold{}.pth'.format(timestr,fold))

# generate submit
submit = reduce(lambda left, right: pd.merge(left, right, on='seg_id'),preds)
submit['time_to_failure'] = submit[['time_to_failure_{}'.format(i) for i in range(cv)]].mean(axis=1)
submit = submit[['time_to_failure']]
assert submit.shape[0] == test_num
validate_mean_error /= cv
submit.to_csv('NN_{}/submit_{:.6f}.csv'.format(timestr,validate_mean_error))
pred_mean = submit['time_to_failure'].mean()
pred_std = submit['time_to_failure'].std()

utils.check_submit(submit)

# generate pred on train
pred_on_train = pd.concat(pred_on_train,ignore_index=True)
pred_on_train = pred_on_train.sort_values(by='idx')
assert pred_on_train['idx'].values.tolist() == list(range(num_samples))
pred_on_train.set_index('idx')
assert pred_on_train.shape == (num_samples,2)
pred_on_train.to_csv('NN_{}/pred_on_train.csv'.format(timestr))

# recrord
record += 'Validate CV Mean: {}\n'.format(validate_mean_error)
record += '\nPred Mean:{}\nPred Std:{}\n'.format(pred_mean,pred_std)
with open('NN_{}/train.log'.format(timestr),'w') as f:
    f.write(record)