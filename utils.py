from global_variable import *
import random
import pickle
import numpy as np
import pandas as pd
import gc
import os
from scipy.stats import ks_2samp

import warnings
warnings.filterwarnings("ignore")

num_samples = 10000

def cross_earthquake(end_index):
    for i in range(intact_wave_num):
        if (end_index - test_length) < earthquake_point[i] and end_index > earthquake_point[i]:
            return True
    return False

def check_data_align(DL_train_data,LGB_train_data,DL_test_data,LGB_test_data):
    from DL import dataset_helper
    from LGB import dataset

    # check train
    print('Check Train Data')
    with open(DL_train_data,'rb') as f:
        dl_train_data = pickle.load(f)
    dl_data = dl_train_data['data']

    lgb_train_data = pd.read_csv(LGB_train_data,index_col=0)
    lgb_data = lgb_train_data.drop(columns=['ttf'], inplace=False)

    with open('sample_end_indexs_20000.pkl','rb') as f:
        sample_end_indexs = pickle.load(f)

    # trainset_path = '/data2/jianglibin/earthquake/train.csv'
    data_set = pd.read_csv(train_csv_path, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    x = data_set['acoustic_data']
    if normalize:
        x = (data_set['acoustic_data'] - acoustic_data_mean) / acoustic_data_std

    check_index = random.sample(list(range(num_samples)),10)
    for i in check_index:
        end_index = sample_end_indexs[i]
        print('index: {}, end position:{}'.format(i,end_index))
        seg = x[end_index-test_length:end_index]
        dl_feature = dataset_helper.create_features2(seg)
        lgb_feature = dataset.create_features(seg)

        print((dl_feature == dl_data[i]).all(), np.abs(pd.Series(lgb_feature).mean()-lgb_data.iloc[i].mean())<1e-5)

    del x
    gc.collect()

    # check test
    print('Check Test Data')
    with open(DL_test_data, 'rb') as f:
        test_data_pkl = pickle.load(f)
    dl_test_data = test_data_pkl['data']
    dl_seg_id = test_data_pkl['seg_id']

    test_data = pd.read_csv(LGB_test_data, index_col=0)
    lgb_seg_id = test_data['seg_id'].values.tolist()
    lgb_test_data = test_data.drop(columns=['seg_id'], inplace=False)
    assert lgb_seg_id==dl_seg_id

    with open('test_file.pkl','rb') as f:
        test_file = pickle.load(f)
    check_index = random.sample(list(range(test_num)),10)
    for i in check_index:
        test_file_name = test_file[i]
        seg = pd.read_csv(os.path.join(test_file_path, test_file_name), dtype={'acoustic_data': np.int16, })['acoustic_data']
        if normalize:
            seg = (seg - acoustic_data_mean) / acoustic_data_std
        lgb_feature = dataset.create_features(seg)
        dl_feature = dataset_helper.create_features2(seg)
        print((dl_feature == dl_test_data[i]).all(), np.abs(pd.Series(lgb_feature).mean()-lgb_test_data.iloc[i].mean())<1e-5)


# 检查submit跟目标submit的差距是否过大
# 目标submit是LB较高的submit
def check_submit(pred_submit,pred_target_path='ensemble_submit/submit_1.95676/submit_1.956760.csv'):
    pred_target = pd.read_csv(pred_target_path, index_col=0)
    # pred_test = pd.read_csv('DL/checkpoints/Earthquake_LSTM_0509-2208/submission_2.008429.csv',index_col=0)
    # pred_test = pd.read_csv('LGB/LGBM_shuffle_0508-0822/submit_1.858939742386934',index_col=0)
    # pred_test = pd.read_csv('NN_0511-1151/submission_1.996231.csv',index_col=0)

    pred_target = pred_target.rename(columns={'time_to_failure': 'ttf_target'})
    pred_test = pred_submit.rename(columns={'time_to_failure': 'ttf_test'})

    d = pd.merge(pred_target, pred_test, on='seg_id')

    d['error'] = np.abs(d['ttf_target'] - d['ttf_test'])
    # print(d['error'].mean())
    print('Check Submit:',d['error'].mean()<1)
    assert d['error'].mean()<1

# 检测特征在train和test的分布是否一致
def feature_distribute_difference(train_path='LGB/train_data_(10000, 909).csv', test_path='LGB/test_data_(2624, 909).csv'):
    train_data = pd.read_csv(train_path, index_col=0).drop(columns=['ttf'], inplace=False)
    test_data = pd.read_csv(test_path, index_col=0).drop(columns=['seg_id'], inplace=False)

    print('Column Check:', (train_data.columns == test_data.columns).all())

    feature_distribute = {}

    for key in train_data.columns:
        x = train_data[key].values
        y = test_data[key].values
        res = ks_2samp(x, y)
        feature_distribute[key] = [res[0], res[1]]  # [feature,[d,p-value]]

    with open('feature_distribute_{}.pkl'.format(train_data.shape[1]+1),'wb') as f:
        pickle.dump(feature_distribute,f)


# 合并手工特征 + DL特征
def load_merge_feature(handcraft_feature_num=100,corr_check=True,corr_threshold=0.9,drop_nainf=False,importance_type='gain',ks_check=False,threshold_d=0.06,threshold_p=1e-6,):
    # 909 : threshold_d=0.025,threshold_p=0.01
    # 921 : threshold_d = 5e-2,threshold_p = 1e-2
    # 2333: threshold_d = 2e-2,threshold_p = 1e-2, top5000
    # load lgb features
    lgb_train_feature_path = '/data2/jianglibin/earthquake/data/LGB/train_data_(10000, 2333).csv'
    lgb_test_features_path = '/data2/jianglibin/earthquake/data/LGB/test_data_(2624, 2333).csv'
    feature_importance = pd.read_csv('LGB/feature_importance_2333.csv',index_col=0)
    assert importance_type=='gain' or importance_type=='split'
    if importance_type=='gain':
        feature_importance = feature_importance.sort_values(by='importance_gain', ascending=False)
        feature_importance = feature_importance[feature_importance['importance_gain']>0]
    else:
        feature_importance = feature_importance.sort_values(by='importance_split', ascending=False)
        feature_importance = feature_importance[feature_importance['importance_split']>0]
    print('usefull feature num:',feature_importance.shape[0])


    num_samples_ = int(lgb_train_feature_path.split(',')[0].split('(')[1])
    extract_feature_num = int(lgb_train_feature_path.split(' ')[1].split(')')[0])

    lgb_train_data = pd.read_csv(lgb_train_feature_path, index_col=0)
    lgb_test_data = pd.read_csv(lgb_test_features_path, index_col=0)

    assert num_samples_ == num_samples
    assert lgb_train_data.shape[0] == num_samples
    assert lgb_train_data.shape[1] == extract_feature_num

    # print('feature_num1:',lgb_train_data.shape[1]-1)

    features = feature_importance['feature'][:min(8000,feature_importance.shape[0])].tolist()
    handcraft_feature_train = pd.DataFrame()
    handcraft_feature_test = pd.DataFrame()

    # generate new feature df
    for mf in features:
        if '+' in mf:
            col1 = mf.split('+')[0].strip()
            col2 = mf.split('+')[1].strip()
            handcraft_feature_train[mf] = lgb_train_data[col1] + lgb_train_data[col2]
            handcraft_feature_test[mf] = lgb_test_data[col1] + lgb_test_data[col2]
        elif '-' in mf:
            col1 = mf.split('-')[0].strip()
            col2 = mf.split('-')[1].strip()
            handcraft_feature_train[mf] = lgb_train_data[col1] - lgb_train_data[col2]
            handcraft_feature_test[mf] = lgb_test_data[col1] - lgb_test_data[col2]
        elif '*' in mf:
            col1 = mf.split('*')[0].strip()
            col2 = mf.split('*')[1].strip()
            handcraft_feature_train[mf] = lgb_train_data[col1] * lgb_train_data[col2]
            handcraft_feature_test[mf] = lgb_test_data[col1] * lgb_test_data[col2]
        elif '/' in mf:
            col1 = mf.split('/')[0].strip()
            col2 = mf.split('/')[1].strip()
            handcraft_feature_train[mf] = lgb_train_data[col1] / lgb_train_data[col2]
            handcraft_feature_test[mf] = lgb_test_data[col1] / lgb_test_data[col2]
        else:
            handcraft_feature_train[mf] = lgb_train_data[mf]
            handcraft_feature_test[mf] = lgb_test_data[mf]
    handcraft_feature_train['ttf'] = lgb_train_data['ttf']
    handcraft_feature_test['seg_id'] = lgb_test_data['seg_id']
    print('feature_num2:', handcraft_feature_train.shape[1]-1)

    del lgb_train_data
    del lgb_test_data
    gc.collect()

    # filter feature,drop different distribute in train_test
    if ks_check:
        feature_distribute = []
        for key in features:
            x = handcraft_feature_train[key].values
            y = handcraft_feature_test[key].values
            res = ks_2samp(x, y)
            feature_distribute.append([key, res[0], res[1]])  # [feature,d,p-value]
        features = list(filter(lambda x: x[1] < threshold_d and x[2] > threshold_p, feature_distribute))
        features = [f[0] for f in features]
        print('feature_num3:',len(features))

    # for f in features:
    #     std_ = handcraft_feature_train[f].std()
    #     if std_ == 0:
    #         print('-------------')
    #         print(f)
    #         print(feature_importance[feature_importance['feature']==f])

    # filter features by corr
    if corr_check:
        features = features[:min(handcraft_feature_num*10,len(features))]
        train_data = handcraft_feature_train[features + ['ttf']]
        corr_matrix = train_data.corr().abs()

        important = features
        corr_matrix = corr_matrix.drop(columns=['ttf'])
        corr_matrix = corr_matrix.drop(index=['ttf'])
        top_feature = []
        for i in range(handcraft_feature_num):
            idx = important[0]
            top_feature.append(idx)
            # print(idx)
            drop_index = list(corr_matrix[idx][corr_matrix[idx] > corr_threshold].index)
            # print(idx, len(drop_index))
            for c in drop_index:
                important.remove(c)
            corr_matrix = corr_matrix.drop(columns=drop_index)
            corr_matrix = corr_matrix.drop(index=drop_index)
    else:
        top_feature = features[:handcraft_feature_num]


    train_features = handcraft_feature_train.drop(columns=['ttf'], inplace=False)[top_feature].values
    train_label = handcraft_feature_train['ttf'].values

    test_features = handcraft_feature_test.drop(columns=['seg_id'], inplace=False)[top_feature].values
    seg_id = handcraft_feature_test['seg_id']

    assert len(seg_id) == test_num
    assert train_features.shape[1] == handcraft_feature_num
    assert train_features.shape[1] == test_features.shape[1]
    print('Handcraft Feature:{}'.format(handcraft_feature_num))

    del handcraft_feature_train
    del handcraft_feature_test
    gc.collect()

    train_features = np.expand_dims(train_features,0).repeat(cv,axis=0)
    test_features = np.expand_dims(test_features,0).repeat(cv,axis=0)

    train_data = [train_features]
    test_data = [test_features]
    feature_num = {'Handcraft Feature':handcraft_feature_num}

    # train_data = []
    # test_data = []
    # feature_num = {}

    # DL_feature_folder = ['Earthquake_LSTM_0521-1713',   # 25 seg
    #                      'Earthquake_LSTM_0521-1854',   # 100 seg
    #                      'Earthquake_CNN_0521-1745',    # 25 seg
    #                      'Earthquake_CNN_0521-1922',    # 100 seg
    #                      ]

    # 203
    # DL_feature_folder = ['Earthquake_LSTM_0522-2358',   # 25 seg
    #                      'Earthquake_LSTM_0523-0917',   # 100 seg
    #                      'Earthquake_CNN_0523-1032',    # 25 seg
    #                      'Earthquake_CNN_0523-1014',    # 100 seg
    #                      ]

    # # 230
    # DL_feature_folder = ['Earthquake_LSTM_0529-2220',
    #                      'Earthquake_CNN_0529-2257',
    #                      'Earthquake_LSTM_0530-1025',
    #                      'Earthquake_CNN_0530-1051',]

    DL_feature_folder = ['Earthquake_LSTM_0531-2049',
                         'Earthquake_CNN_0531-2158',
                         'Earthquake_LSTM_0531-2316',
                         'Earthquake_CNN_0531-2231',]
    # DL_feature_folder = []

    for folder in DL_feature_folder:
        train_feature = np.load(os.path.join('DL/checkpoints',folder,'train_features_(5, 10000, 50).npy'))
        test_feature = np.load(os.path.join('DL/checkpoints', folder, 'test_features_(5, 2624, 50).npy'))
        feature_num_folder = train_feature.shape[2]
        feature_num[folder] = feature_num_folder
        print('{}:{}'.format(folder,feature_num_folder))

        train_data.append(train_feature)
        test_data.append(test_feature)
    train_data = np.concatenate(train_data,axis=2)
    test_data = np.concatenate(test_data,axis=2)
    feature_num['Total num'] = sum([v for k,v in feature_num.items()])
    print('Total num:',feature_num['Total num'])


    # remove inf
    if drop_nainf:
        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)
        train_data = np.clip(train_data,-1e10,1e10)
        test_data = np.clip(test_data,-1e10,1e10)

    # normalize parameter
    data_mean = train_data.mean(axis=0).mean(axis=0)
    data_std = train_data.mean(axis=0).std(axis=0)
    label_mean = train_label.mean(axis=0)
    label_std = train_label.std(axis=0)

    assert data_mean.shape == (feature_num['Total num'],)
    assert data_std.shape == (feature_num['Total num'],)
    assert (data_std==0).sum() == 0

    # print(np.isnan(train_data).sum())
    # print(np.isinf(train_data).sum())
    # print(np.isnan(test_data).sum())
    # print(np.isinf(test_data).sum())

    if drop_nainf:
        assert np.isnan(train_data).sum() == 0
        assert np.isnan(test_data).sum() == 0
        assert np.isinf(train_data).sum() == 0
        assert np.isinf(test_data).sum() == 0

        assert np.isnan(data_mean).sum() == 0
        assert np.isinf(data_mean).sum() == 0

    assert train_data.shape == (cv,num_samples,feature_num['Total num'])
    assert test_data.shape == (cv,test_num,feature_num['Total num'])

    return train_data,train_label,test_data,seg_id,data_mean,data_std,label_mean,label_std,feature_num#,lgb_train_features_df


if __name__ == '__main__':
    DL_train_data = 'DL/train_data_(20000, 25, 95).pkl'
    LGB_train_data = 'LGB/train_data_(20000, 879).csv'
    DL_test_data = 'DL/test_data_(2624, 25, 95).pkl'
    LGB_test_data = 'LGB/test_data_(2624, 879).csv'

    # check_data_align(DL_train_data,LGB_train_data,DL_test_data,LGB_test_data)

    # load_merge_feature(100)

    # feature_distribute_difference()
    x = load_merge_feature(20,ks_check=True,corr_check=True)
