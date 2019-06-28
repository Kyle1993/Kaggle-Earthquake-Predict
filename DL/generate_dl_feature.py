import pandas as pd
import numpy as np
import sys,os
import torch
from functools import reduce
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
from scipy.stats import ks_2samp
import time
from torch import optim
from sklearn.metrics import mean_absolute_error


sys.path.append('../')
from DL import models
from DL.config import *
from DL.dataset_helper import EarthQuakeDataset_test,EarthQuakeDataset
from DL import dl_utils
import utils

config.feature_num = 230
config.seg_size = (100,1500)
config.ks_check = False
config.extract_feature_num = 10
model_name = 'Earthquake_CNN'

config.lr = 1e-4
config.epoch = 10

# # if CNN & (100,1500)
# config.lr = 1e-3r
# config.epoch = 6

with open('../kfold_{}.pkl'.format(num_samples),'rb') as f:
    kfold = pickle.load(f)

with open('../test_file.pkl','rb') as f:
    test_file = pickle.load(f)
test_file = [tf.split('.')[0] for tf in test_file]


with open('/data2/jianglibin/earthquake/data/DL/test_data_(2624, {}, {}).pkl'.format(config.seg_size[0],config.feature_num),'rb') as f:
    test_data_pkl = pickle.load(f)
test_data = test_data_pkl['data']
seg_id = test_data_pkl['seg_id']

assert seg_id == test_file

with open('/data2/jianglibin/earthquake/data/DL/train_data_({}, {}, {}).pkl'.format(num_samples,config.seg_size[0],config.feature_num),'rb') as f:
    train_data = pickle.load(f)
train_label = train_data['label']
train_data = train_data['data']
assert train_data.shape[0] == num_samples

# ks-check
if config.ks_check:
    train_data_ = train_data.reshape(-1,config.feature_num)
    test_data_ = test_data.reshape(-1,config.feature_num)
    save_columns = []
    for i in range(config.feature_num):
        x = train_data_[:,i]
        y = test_data_[:,i]
        res = ks_2samp(x, y)
        d_value = res[0]
        p_value = res[1]
        if (d_value<config.threshold_d and p_value>config.threshold_p):
            save_columns.append(i)

    feature_num_before = train_data.shape[2]
    train_data = train_data[:,:,save_columns]
    test_data = test_data[:,:,save_columns]
    feature_num_after = train_data.shape[2]

    print('feature num before ks-filte:',feature_num_before)
    print('feature num after ks-filte:',feature_num_after)

config.feature_num = train_data.shape[2]

data_mean = train_data.mean(axis=0)
data_std = train_data.std(axis=0)
label_mean = train_label.mean(axis=0)
label_std = train_label.std(axis=0)

test_data = (test_data - data_mean) / (data_std+1)
train_data = (train_data - data_mean) / (data_std+1)
train_label = (train_label - label_mean) / label_std

timestr = time.strftime('%m%d-%H%M')
# makedir
if not os.path.exists('checkpoints/{}_{}'.format(model_name, timestr)):
    os.mkdir('checkpoints/{}_{}'.format(model_name, timestr))

validate_mean_error = 0
record = 'Time:{}\n\n'.format(timestr)
record += 'Sample Num:{}\n'.format(num_samples)
record += config.to_str()
record += '\n{:<5}\t{:<20}\t{:<20}\n'.format('Fold','Train Error','Validate Error')

preds = []
train_features = []
test_features = []

for fold in range(cv):
    print('Fold {}:'.format(fold))
    train_idxs = kfold[fold][0]
    validate_idxs = kfold[fold][1]

    model = models.load_model(model_name,input_size=config.feature_num,output_size=config.extract_feature_num,drop=True)
    if config.gpu>=0:
        if 'CNN' in model.__class__.__name__:
            model.cuda_(config.gpu)
        else:
            model.cuda(config.gpu)
    criticer = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=config.lr)

    # training
    print('training...')
    model.train()
    train_y = []
    train_pred = []
    for epoch in tqdm(range(config.epoch)):
        trainset = EarthQuakeDataset(train_data, train_label, train_idxs)
        trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

        for _,traindata,trainlabel in trainloader:
            traindata = dl_utils.totensor(traindata).float()
            trainlabel = dl_utils.totensor(trainlabel).float()
            pred,_ = model(traindata)
            if epoch == config.epoch -1:
                train_y.append(trainlabel.data.cpu().numpy())
                train_pred.append(pred.data.cpu().numpy())
            loss = criticer(pred,trainlabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if epoch == config.epoch_decay:
        #     optimizer = optim.Adam(model.parameters(), lr=config.lr/20)

    train_y = np.concatenate(train_y, axis=0)
    train_pred = np.concatenate(train_pred, axis=0)
    train_error = mean_absolute_error(train_y, train_pred) * label_std


    # vlaidate
    print('validating...')
    model.eval()
    validateset = EarthQuakeDataset(train_data, train_label, validate_idxs)
    validateloader = DataLoader(validateset, batch_size=config.validate_batch_size, shuffle=True)

    validate_y = []
    validate_pred = []
    with torch.no_grad():
        for _, validatedata, validatelabel in tqdm(validateloader):
            validatedata = dl_utils.totensor(validatedata).float()
            pred, _ = model(validatedata)
            validate_pred.append(pred.cpu().numpy())
            validate_y.append(validatelabel)
    validate_y = np.concatenate(validate_y, axis=0)
    validate_pred = np.concatenate(validate_pred, axis=0)
    validate_error = mean_absolute_error(validate_y, validate_pred) * label_std

    print('Train Error:{:.6f}\tValidate Error:{:.6f}'.format(train_error,validate_error))
    record += '{:<5}\t{:<20}\t{:<20}\n'.format(fold,train_error,validate_error)
    validate_mean_error += validate_error

    # get DL features & inference
    print('getting DL features & inference...')
    trainset = EarthQuakeDataset(train_data,train_label,list(range(num_samples)))
    trainloader = DataLoader(trainset,batch_size=64,shuffle=False)
    testset = EarthQuakeDataset_test(test_data,seg_id)
    testloader = DataLoader(testset,batch_size=64,shuffle=False)

    model.eval()
    with torch.no_grad():
        train_features_fold = []
        for _,traindata,trainlabel in tqdm(trainloader):
            traindata = dl_utils.totensor(traindata).float()
            _,trainfeature = model(traindata)
            train_features_fold.append(trainfeature.cpu().numpy())
        train_features_fold = np.concatenate(train_features_fold,axis=0)

        test_names = []
        test_features_fold = []
        testpreds = []
        for testdata, testname in tqdm(testloader):
            testdata = dl_utils.totensor(testdata).float()
            testpred,testfeature = model(testdata)
            testpreds.append(testpred.cpu().numpy())

            test_features_fold.append(testfeature.cpu().numpy())
            test_names.extend(testname)

        test_features_fold = np.concatenate(test_features_fold,axis=0)

    assert train_features_fold.shape[0] == num_samples
    assert test_features_fold.shape[0] == test_num
    assert test_names == test_file

    testpreds = np.concatenate(testpreds,axis=0)
    testpreds = testpreds*label_std + label_mean
    testpreds = np.clip(testpreds,0,20)

    preds_fold = pd.DataFrame({'seg_id':test_names,'time_to_failure_{}'.format(fold):testpreds})
    preds_fold.set_index('seg_id',drop=True,inplace=True)
    preds.append(preds_fold)

    train_features.append(train_features_fold)
    test_features.append(test_features_fold)

    # # save fold model
    # torch.save(model.train().cpu().state_dict(),'checkpoints/{}_{}/fold{}.pth'.format(model_name,timestr,fold))


validate_mean_error /= cv
record += 'Validate CV Mean: {}\n'.format(validate_mean_error)

# generate submit
submit = reduce(lambda left, right: pd.merge(left, right, on='seg_id'),preds)
submit['time_to_failure'] = submit[['time_to_failure_{}'.format(i) for i in range(cv)]].mean(axis=1)
submit = submit[['time_to_failure']]
assert submit.shape[0] == test_num
# utils.check_submit(submit,'checkpoints/Earthquake_LSTM_0510-1129/submission_2.017596.csv')
submit.to_csv('checkpoints/{}_{}/submit_{:.6f}.csv'.format(model_name,timestr,validate_mean_error))

pred_mean = submit['time_to_failure'].mean()
pred_std = submit['time_to_failure'].std()

# save record
record += '\nPred Mean:{}\nPred Std:{}\n'.format(pred_mean,pred_std)
with open('checkpoints/{}_{}/train.log'.format(model_name, timestr),'w') as f:
    f.write(record)

# save features
train_features = np.asarray(train_features)
test_features = np.asarray(test_features)

np.save('checkpoints/{}_{}/train_features_{}.npy'.format(model_name,timestr,train_features.shape),train_features)
np.save('checkpoints/{}_{}/test_features_{}.npy'.format(model_name,timestr,test_features.shape),test_features)








