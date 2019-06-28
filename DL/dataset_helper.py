import pandas as pd
import numpy as np
import sys,os
from multiprocessing import Process,Manager
from tsfresh.feature_extraction import feature_calculators
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold
import scipy.signal as sg
from scipy import stats
from scipy.signal.windows import hann
from scipy.signal import hilbert
from scipy.signal import convolve
from torch.utils.data import Dataset,DataLoader
import scipy.fftpack as sf
import gc

sys.path.append('../')
from DL.config import *


'''
*discuss里有人说波形可能发生在两次earthquake之间

1.加大seg的长度,缩小间隔  ： 不敏感,用25就不错
2.简化seg的feature_create: 加上rolling效果很好
3.RNN对Adam,SGD不敏感, CNN要用Adam
4.CNN要用relu6，用relu爆炸,用sigmoid运算太慢

feature engineering:
1.加abs:有一点提升
2.change & change rate:有一点提升
3.step == 5000时降低学习率： 待定
4.加fft：效果不明显,运算慢了很多

5.加指数加权平均：目测没啥用
6.加中位数：已添加
7.不同窗口的rolling:在50的基础上加了500，效果不明显
8.添加miniseg,也就是最后10%和20%的数据的统计：有大约0.1的提升，后面考虑添加更多的？？
9.添加peak10:有一点提升
9.使用adabound:效果不行
10.数据标准化：开始训练加速收敛，但对后期收敛效果影响不大

11.模型结构：
12.使用cnn:效果不如rnn但是可以后期用来做模型融合
13.用log-cosh loss


'''


MAX_FREQ_IDX_DL = config.seg_size[1]
FREQ_STEP_DL = int(MAX_FREQ_IDX_DL/2)


def count_peak(seg,k):
    res = [feature_calculators.number_peaks(seg[i], k) for i in range(config.seg_size[0])]
    return np.asarray(res)

def autocorrelation(seg,k):
    res = [feature_calculators.autocorrelation(seg[i],k) for i in range(config.seg_size[0])]
    return np.asarray(res)

def ratio_value_number_to_time_series_length(seg):
    res = [feature_calculators.ratio_value_number_to_time_series_length(seg[i]) for i in range(config.seg_size[0])]
    return np.asarray(res)

def cross_0(seg):
    res = [feature_calculators.number_crossing_m(seg[i],0) for i in range(config.seg_size[0])]
    return np.asarray(res)

def longest_strike_above_mean(seg):
    res = [feature_calculators.longest_strike_above_mean(seg[i]) for i in range(config.seg_size[0])]
    return np.asarray(res)

def binned_entropy(seg,k):
    res = [feature_calculators.binned_entropy(seg[i],k) for i in range(config.seg_size[0])]
    return np.asarray(res)

def create_features2(seg, ):
    data_row = {}
    seg = seg.values.reshape(config.seg_size)

    zc = sf.fft(seg, axis=1)

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    magFFT = np.abs(zc)
    phzFFT = np.angle(zc)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    for freq in range(0, MAX_FREQ_IDX_DL, FREQ_STEP_DL):
        data_row['FFT_Mag_10q%d' % freq] = np.quantile(magFFT[:, freq: freq + FREQ_STEP_DL], 0.1, axis=1)
        data_row['FFT_Mag_90q%d' % freq] = np.quantile(magFFT[:, freq: freq + FREQ_STEP_DL], 0.9, axis=1)
        data_row['FFT_Mag_mean%d' % freq] = np.mean(magFFT[:, freq: freq + FREQ_STEP_DL], axis=1)
        data_row['FFT_Mag_std%d' % freq] = np.std(magFFT[:, freq: freq + FREQ_STEP_DL], axis=1)
        data_row['FFT_Mag_max%d' % freq] = np.max(magFFT[:, freq: freq + FREQ_STEP_DL], axis=1)

        data_row['FFT_Phz_mean%d' % freq] = np.mean(phzFFT[:, freq: freq + FREQ_STEP_DL], axis=1)
        data_row['FFT_Phz_std%d' % freq] = np.std(phzFFT[:, freq: freq + FREQ_STEP_DL], axis=1)

    seg = pd.DataFrame(seg)
    sigs = [seg]

    sigs.append(pd.DataFrame(realFFT))
    sigs.append(pd.DataFrame(imagFFT))

    for span in [30]:
        exp_mean = seg.ewm(span,axis=1).mean().dropna(axis=1)
        exp_std = seg.ewm(span,axis=1).std().dropna(axis=1)
        sigs.append(exp_mean)
        sigs.append(exp_std)
    #
    for w in [50,500]:
        roll_std = seg.rolling(w, axis=1).std().dropna(axis=1)
        roll_mean = seg.rolling(w, axis=1).mean().dropna(axis=1)
        sigs.append(roll_mean)
        sigs.append(roll_std)

    for i,sig in enumerate(sigs):
        data_row['mean_%d' % i] = sig.mean(axis=1)
        data_row['std_%d' % i] = sig.std(axis=1)
        data_row['max_%d' % i] = sig.max(axis=1)
        data_row['min_%d' % i] = sig.min(axis=1)
        data_row['q09_%d' % i] = np.quantile(sig, 0.9, axis=1)
        data_row['q05_%d' % i] = np.quantile(sig, 0.5, axis=1)
        data_row['q01_%d' % i] = np.quantile(sig, 0.1, axis=1)
        data_row['abs_q09_%d' % i] = np.quantile(np.abs(sig), 0.9, axis=1)
        data_row['abs_q05_%d' % i] = np.quantile(np.abs(sig), 0.5, axis=1)
        data_row['abs_q01_%d' % i] = np.quantile(np.abs(sig), 0.1, axis=1)
        data_row['av_change_abs_%d' % i] = np.mean(np.diff(sig, axis=1), axis=1)
        data_row['peak10_k{}_{}'.format(10,i)] = count_peak(np.asarray(sig),10)
        data_row['autocorrelation_k{}_{}'.format(5,i)] = autocorrelation(np.asarray(sig),5)
        data_row['autocorrelation_k{}_{}'.format(50,i)] = autocorrelation(np.asarray(sig),50)
        data_row['ratio_value_number_%d' % i] = ratio_value_number_to_time_series_length(np.asarray(sig))
        data_row['cross_0_%d' % i] = cross_0(np.asarray(sig))
        data_row['binned_entropy_k{}_{}'.format(10,i)] = binned_entropy(np.asarray(sig),10)
        data_row['binned_entropy_k{}_{}'.format(100,i)] = binned_entropy(np.asarray(sig),100)


        data_row['mad_%d' % i] = sig.mad(axis=1)
        data_row['kurt_%d' % i] = sig.kurtosis(axis=1)
        data_row['skew_%d' % i] = sig.skew(axis=1)
        data_row['med_%d' % i] = sig.median(axis=1)

        w = 20
        signal_mean = sig.rolling(window=w,axis=1).mean().dropna(axis=1)
        signal_std = sig.rolling(window=w,axis=1).std().dropna(axis=1)
        data_row['high_bound_mean_win{}_{}'.format(w, i)] = (signal_mean + 2 * signal_std).mean(axis=1)
        data_row['low_bound_mean_win{}_{}'.format(w, i)] = (signal_mean - 2 * signal_std).mean(axis=1)

    data_row = [d[1] for d in sorted(data_row.items(), key=lambda x: x[0])]
    data_row = np.asarray(data_row)
    return data_row.transpose()

def load_train_multiprocess(l,x,y,end_indexs,pid):
    X = []
    Y = []
    y = y.values

    load_index = [] # for data align check
    for sample_end_index in tqdm(end_indexs,postfix=pid):
        seg = x[(sample_end_index - test_length):sample_end_index]
        X.append(create_features2(seg))
        Y.append(y[sample_end_index])
        load_index.append(sample_end_index)

    X = np.asarray(X)
    Y = np.asarray(Y)

    l.append((pid,X,Y,load_index),)

def load_train_data(train_csv_path,sample_end_indexs,num_worker=5):
    num = len(sample_end_indexs)
    assert num % num_worker == 0

    dataset = pd.read_csv(train_csv_path, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    x = dataset['acoustic_data']
    y = dataset['time_to_failure']
    # if normalize:
    #     x = (dataset['acoustic_data'] - acoustic_data_mean) / acoustic_data_std
    print('CSV Data Loaded!')

    # load train data
    num_each_worker = int(num / num_worker)
    end_index_each_worker = []
    for i in range(num_worker):
        if i < (num_worker - 1):
            end_index_each_worker.append(sample_end_indexs[i * num_each_worker:(i + 1) * num_each_worker])
        else:
            end_index_each_worker.append(sample_end_indexs[i * num_each_worker:])

    with Manager() as manager:
        l_process = manager.list()
        workers = [Process(target=load_train_multiprocess, args=(l_process, x, y, end_index_each_worker[pid], pid)) for
                   pid in range(num_worker)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        l = list(l_process)
        l = sorted(l, key=lambda x: x[0])

    data = []
    label = []
    data_index = []
    for d in l:
        data.append(d[1])
        label.append(d[2])
        data_index.extend(d[3])
    data = np.concatenate(data,axis=0)
    label = np.concatenate(label,axis=0)
    assert data.shape[0] == num_samples
    assert label.shape[0] == num_samples
    assert data_index == sample_end_indexs # check data align
    print('Data Process Done!({} samples, shape:{})'.format(num_samples, data.shape))

    return data,label

def load_test_multiprocess(l,test_path,test_file_list,pid):
    X = []

    for f in tqdm(test_file_list,postfix=pid):
        seg = pd.read_csv(os.path.join(test_path, f), dtype={'acoustic_data': np.int16, })['acoustic_data']
        # if normalize:
        #     seg = (seg - acoustic_data_mean) / acoustic_data_std
        segment = create_features2(seg)
        X.append(segment)

    X = np.asarray(X)
    seg_id = [tf.split('.')[0] for tf in test_file_list]

    l.append((pid,X,seg_id),)

def load_test_data(num_worker=5,):
    with open('../test_file.pkl','rb') as f:
        test_file = pickle.load(f)
    assert len(test_file) == test_num
    num_each_worker = int(len(test_file)/num_worker)
    test_file_each_worker = []
    for i in range(num_worker):
        if i < (num_worker-1):
            test_file_each_worker.append(test_file[i*num_each_worker:(i+1)*num_each_worker])
        else:
            test_file_each_worker.append(test_file[i*num_each_worker:])
    l = []
    with Manager() as manager:
        l_process = manager.list()
        workers = [Process(target=load_test_multiprocess, args=(l_process,test_file_path,test_file_each_worker[pid], pid)) for pid in range(num_worker)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        l = list(l_process)
        l = sorted(l, key=lambda x: x[0])

    test_data = []
    seg_id = []
    for d in l:
        test_data.append(d[1])
        seg_id.extend(d[2])
    test_data = np.concatenate(test_data,axis=0)

    assert len(seg_id) == len(set(seg_id))
    assert test_data.shape[0] == test_num
    assert len(seg_id) == test_num

    # check data align
    test_file = [tf.split('.')[0] for tf in test_file]
    assert seg_id == test_file
    print('Test Data Loaded!(2624 samples, shape:{})'.format(test_data.shape))

    return test_data,seg_id


class EarthQuakeDataset(Dataset):
    def __init__(self,data,label,idxs,):
        super(EarthQuakeDataset,self).__init__()
        self.idxs = idxs
        if isinstance(data,np.ndarray):
            self.data = data[idxs]
            self.label = label[idxs]
        else:
            self.data = data.iloc[idxs]
            self.label = data.iloc[idxs]

    def __getitem__(self, idx):
        return self.idxs[idx],self.data[idx],self.label[idx]

    def __len__(self):
        return len(self.idxs)

class EarthQuakeDataset_test(Dataset):
    def __init__(self,test_data,seg_id):
        super(EarthQuakeDataset_test,self).__init__()
        self.data = test_data
        self.seg_id = seg_id

    def __getitem__(self, idx):
        return self.data[idx],self.seg_id[idx]

    def __len__(self):
        return len(self.seg_id)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    with open('../sample_end_indexs_{}.pkl'.format(num_samples), 'rb') as f:
        sample_end_indexs = pickle.load(f)
    assert len(sample_end_indexs) == num_samples

    train, label = load_train_data(train_csv_path, sample_end_indexs, num_worker=5)
    print(train.shape)
    print(label.shape)

    train_data = {'data':train,'label':label}
    with open('/data2/jianglibin/earthquake/data/DL/train_data_{}.pkl'.format(train.shape),'wb') as f:
        pickle.dump(train_data,f)

    data_mean = train.mean(axis=0)
    data_std = train.std(axis=0)
    label_mean = label.mean(axis=0)
    label_std = label.std(axis=0)

    data_normalize = {'data_mean': data_mean, 'data_std': data_std, 'label_mean': label_mean, 'label_std': label_std}
    with open('/data2/jianglibin/earthquake/data/DL/data_normalize_{}.pkl'.format(train.shape), 'wb') as f:
        pickle.dump(data_normalize, f)

    test, seg_id = load_test_data(num_worker=5)
    print(test.shape)
    print(len(seg_id))

    test_data = {'data':test,'seg_id':seg_id}
    with open('/data2/jianglibin/earthquake/data/DL/test_data_{}.pkl'.format(test.shape), 'wb') as f:
        pickle.dump(test_data, f)