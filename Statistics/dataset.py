import pandas as pd
import numpy as np
import random
import sys,os
from sklearn.linear_model import LinearRegression
from multiprocessing import Process,Manager
from tsfresh.feature_extraction import feature_calculators
from tqdm import tqdm
import time
import pickle
import scipy.signal as sg
from scipy import stats
from scipy.signal.windows import hann
from scipy.signal import hilbert
from scipy.signal import convolve
import gc

sys.path.append('../')

from global_variable import *
import utils

import warnings
warnings.filterwarnings("ignore")

NY_FREQ_IDX = 75000
CUTOFF = 20000
MAX_FREQ = 20000
FREQ_STEP = 5000

def cross_earthquake(end_index):
    for i in range(intact_wave_num):
        if (end_index - test_length) < earthquake_point[i] and end_index > earthquake_point[i]:
            return True
    return False

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

def des_filter(x, low=None, high=None, order=5):
    if low is None and high is None:
        return
    elif high is None:
        b, a = sg.butter(order, Wn=low / NY_FREQ_IDX, btype='highpass')
    elif low is None:
        b, a = sg.butter(order, Wn=high / NY_FREQ_IDX, btype='lowpass')
    else:
        b, a = sg.butter(order, Wn=(low / NY_FREQ_IDX, high / NY_FREQ_IDX), btype='bandpass')
    sig = sg.lfilter(b, a, x)
    return sig

def create_features(seg, ):
    data_row = {}

    xcz = des_filter(seg, high=CUTOFF)

    zc = np.fft.fft(xcz)
    zc = zc[:MAX_FREQ]

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    freq_bands = list(range(0, MAX_FREQ, FREQ_STEP))
    magFFT = np.abs(zc)
    phzFFT = np.angle(zc)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    for freq in freq_bands:
        data_row['FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)
        data_row['FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)
        data_row['FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)
        data_row['FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)
        data_row['FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_STEP])
        data_row['FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_STEP])
        data_row['FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_STEP])

        data_row['FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_STEP])
        data_row['FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_STEP])

    data_row['FFT_Rmean'] = realFFT.mean()
    data_row['FFT_Rstd'] = realFFT.std()
    data_row['FFT_Rmax'] = realFFT.max()
    data_row['FFT_Rmin'] = realFFT.min()
    data_row['FFT_Imean'] = imagFFT.mean()
    data_row['FFT_Istd'] = imagFFT.std()
    data_row['FFT_Imax'] = imagFFT.max()
    data_row['FFT_Imin'] = imagFFT.min()

    data_row['FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    data_row['FFT_Rstd__first_6000'] = realFFT[:6000].std()
    data_row['FFT_Rmax_first_6000'] = realFFT[:6000].max()
    data_row['FFT_Rmin_first_6000'] = realFFT[:6000].min()
    data_row['FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    data_row['FFT_Rstd_first_18000'] = realFFT[:18000].std()
    data_row['FFT_Rmax_first_18000'] = realFFT[:18000].max()
    data_row['FFT_Rmin_first_18000'] = realFFT[:18000].min()

    del xcz
    del zc
    gc.collect()

    sigs = [seg]
    for freq in range(0,MAX_FREQ+FREQ_STEP,FREQ_STEP):
        if freq==0:
            xc_ = des_filter(seg, high=FREQ_STEP)
        elif freq==MAX_FREQ:
            xc_ = des_filter(seg, low=freq)
        else:
            xc_ = des_filter(seg, low=freq, high=freq+FREQ_STEP)
        sigs.append(pd.Series(xc_))

    for i, sig in enumerate(sigs):
        data_row['mean_%d' % i] = sig.mean()
        data_row['std_%d' % i] = sig.std()
        data_row['max_%d' % i] = sig.max()
        data_row['min_%d' % i] = sig.min()

        data_row['mean_change_abs_%d' % i] = np.mean(np.diff(sig))
        data_row['mean_change_rate_%d' % i] = np.mean(np.nonzero((np.diff(sig) / sig[:-1]))[0])
        data_row['abs_max_%d' % i] = np.abs(sig).max()
        data_row['abs_min_%d' % i] = np.abs(sig).min()

        data_row['std_first_50000_%d' % i] = sig[:50000].std()
        data_row['std_last_50000_%d' % i] = sig[-50000:].std()
        data_row['std_first_10000_%d' % i] = sig[:10000].std()
        data_row['std_last_10000_%d' % i] = sig[-10000:].std()

        data_row['avg_first_50000_%d' % i] = sig[:50000].mean()
        data_row['avg_last_50000_%d' % i] = sig[-50000:].mean()
        data_row['avg_first_10000_%d' % i] = sig[:10000].mean()
        data_row['avg_last_10000_%d' % i] = sig[-10000:].mean()

        data_row['min_first_50000_%d' % i] = sig[:50000].min()
        data_row['min_last_50000_%d' % i] = sig[-50000:].min()
        data_row['min_first_10000_%d' % i] = sig[:10000].min()
        data_row['min_last_10000_%d' % i] = sig[-10000:].min()

        data_row['max_first_50000_%d' % i] = sig[:50000].max()
        data_row['max_last_50000_%d' % i] = sig[-50000:].max()
        data_row['max_first_10000_%d' % i] = sig[:10000].max()
        data_row['max_last_10000_%d' % i] = sig[-10000:].max()

        data_row['max_to_min_%d' % i] = sig.max() / np.abs(sig.min())
        data_row['max_to_min_diff_%d' % i] = sig.max() - np.abs(sig.min())
        data_row['count_big_%d' % i] = len(sig[np.abs(sig) > 500])
        data_row['sum_%d' % i] = sig.sum()

        data_row['mean_change_rate_first_50000_%d' % i] = np.mean(
            np.nonzero((np.diff(sig[:50000]) / sig[:50000][:-1]))[0])
        data_row['mean_change_rate_last_50000_%d' % i] = np.mean(
            np.nonzero((np.diff(sig[-50000:]) / sig[-50000:][:-1]))[0])
        data_row['mean_change_rate_first_10000_%d' % i] = np.mean(
            np.nonzero((np.diff(sig[:10000]) / sig[:10000][:-1]))[0])
        data_row['mean_change_rate_last_10000_%d' % i] = np.mean(
            np.nonzero((np.diff(sig[-10000:]) / sig[-10000:][:-1]))[0])

        data_row['q95_%d' % i] = np.quantile(sig, 0.95)
        data_row['q99_%d' % i] = np.quantile(sig, 0.99)
        data_row['q05_%d' % i] = np.quantile(sig, 0.05)
        data_row['q01_%d' % i] = np.quantile(sig, 0.01)

        data_row['abs_q95_%d' % i] = np.quantile(np.abs(sig), 0.95)
        data_row['abs_q99_%d' % i] = np.quantile(np.abs(sig), 0.99)
        data_row['abs_q05_%d' % i] = np.quantile(np.abs(sig), 0.05)
        data_row['abs_q01_%d' % i] = np.quantile(np.abs(sig), 0.01)

        data_row['trend_%d' % i] = add_trend_feature(sig)
        data_row['abs_trend_%d' % i] = add_trend_feature(sig, abs_values=True)
        data_row['abs_mean_%d' % i] = np.abs(sig).mean()
        data_row['abs_std_%d' % i] = np.abs(sig).std()

        data_row['mad_%d' % i] = sig.mad()
        data_row['kurt_%d' % i] = sig.kurtosis()
        data_row['skew_%d' % i] = sig.skew()
        data_row['med_%d' % i] = sig.median()

        data_row['Hilbert_mean_%d' % i] = np.abs(hilbert(sig)).mean()
        data_row['Hann_window_mean'] = (convolve(seg, hann(150), mode='same') / sum(hann(150))).mean()

        data_row['classic_sta_lta1_mean_%d' % i] = classic_sta_lta(sig, 500, 10000).mean()
        data_row['classic_sta_lta2_mean_%d' % i] = classic_sta_lta(sig, 5000, 100000).mean()
        data_row['classic_sta_lta3_mean_%d' % i] = classic_sta_lta(sig, 3333, 6666).mean()
        data_row['classic_sta_lta4_mean_%d' % i] = classic_sta_lta(sig, 10000, 25000).mean()

        data_row['Moving_average_400_mean_%d' % i] = sig.rolling(window=400).mean().mean(skipna=True)
        data_row['Moving_average_700_mean_%d' % i] = sig.rolling(window=700).mean().mean(skipna=True)
        data_row['Moving_average_1500_mean_%d' % i] = sig.rolling(window=1500).mean().mean(skipna=True)
        data_row['Moving_average_3000_mean_%d' % i] = sig.rolling(window=3000).mean().mean(skipna=True)
        data_row['Moving_average_6000_mean_%d' % i] = sig.rolling(window=6000).mean().mean(skipna=True)

        ewma = pd.Series.ewm
        data_row['exp_Moving_average_300_mean_%d' % i] = ewma(sig, span=300).mean().mean(skipna=True)
        data_row['exp_Moving_average_3000_mean_%d' % i] = ewma(sig, span=3000).mean().mean(skipna=True)
        data_row['exp_Moving_average_30000_mean_%d' % i] = ewma(sig, span=6000).mean().mean(skipna=True)

        no_of_std = 2
        data_row['MA_700MA_std_mean_%d' % i] = sig.rolling(window=700).std().mean(skipna=True)
        data_row['MA_700MA_BB_high_mean_%d' % i] = (
        data_row['Moving_average_700_mean_%d' % i] + no_of_std * data_row['MA_700MA_std_mean_%d' % i]).mean()
        data_row['MA_700MA_BB_low_mean_%d' % i] = (
        data_row['Moving_average_700_mean_%d' % i] - no_of_std * data_row['MA_700MA_std_mean_%d' % i]).mean()
        data_row['MA_400MA_std_mean_%d' % i] = sig.rolling(window=400).std().mean(skipna=True)
        data_row['MA_400MA_BB_high_mean_%d' % i] = (
        data_row['Moving_average_400_mean_%d' % i] + no_of_std * data_row['MA_400MA_std_mean_%d' % i]).mean()
        data_row['MA_400MA_BB_low_mean_%d' % i] = (
        data_row['Moving_average_400_mean_%d' % i] - no_of_std * data_row['MA_400MA_std_mean_%d' % i]).mean()

        data_row['iqr0_%d' % i] = np.subtract(*np.percentile(sig, [75, 25]))
        data_row['q999_%d' % i] = np.quantile(sig, 0.999)
        data_row['q001_%d' % i] = np.quantile(sig, 0.001)
        data_row['ave10_%d' % i] = stats.trim_mean(sig, 0.1)
        data_row['peak10_num_%d' % i] = feature_calculators.number_peaks(sig, 10)
        data_row['num_cross_0_%d' % i] = feature_calculators.number_crossing_m(sig, 0)
        data_row['autocorrelation_%d' % i] = feature_calculators.autocorrelation(sig, 5)
        # data_row['spkt_welch_density_%d' % i] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 50}]))[0][1]
        data_row['ratio_value_number_%d' % i] = feature_calculators.ratio_value_number_to_time_series_length(sig)

    for windows in [50, 200, 1000]:
        x_roll_std = seg.rolling(windows).std().dropna().values
        x_roll_mean = seg.rolling(windows).mean().dropna().values

        data_row['ave_roll_std_' + str(windows)] = x_roll_std.mean()
        data_row['std_roll_std_' + str(windows)] = x_roll_std.std()
        data_row['max_roll_std_' + str(windows)] = x_roll_std.max()
        data_row['min_roll_std_' + str(windows)] = x_roll_std.min()
        data_row['q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        data_row['q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        data_row['q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        data_row['q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        data_row['av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        data_row['av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        data_row['abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        data_row['ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        data_row['std_roll_mean_' + str(windows)] = x_roll_mean.std()
        data_row['max_roll_mean_' + str(windows)] = x_roll_mean.max()
        data_row['min_roll_mean_' + str(windows)] = x_roll_mean.min()
        data_row['q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        data_row['q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        data_row['q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        data_row['q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        data_row['av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        data_row['av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        data_row['abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

        data_row['num_peak10_rolling_' + str(windows)] = feature_calculators.number_peaks(x_roll_mean, 10)
        data_row['num_cross0_rolling_' + str(windows)] = feature_calculators.number_crossing_m(x_roll_mean, 0)
        data_row['autocorrelation_rolling_' + str(windows)] = feature_calculators.autocorrelation(x_roll_mean, 5)
        # data_row['spkt_welch_density_rolling_' + str(windows)] = list(feature_calculators.spkt_welch_density(x_roll_mean, [{'coeff': 50}]))[0][1]
        data_row['ratio_value_number_rolling_' + str(windows)] = feature_calculators.ratio_value_number_to_time_series_length(x_roll_mean)
        data_row['classic_sta_lta_rolling_' + str(windows)] = classic_sta_lta(x_roll_mean, 500, 10000).mean()

    return data_row

def create_features2(seg, ):
    data_row = {}

    xcz = des_filter(seg, high=CUTOFF)

    zc = np.fft.fft(xcz)
    zc = zc[:MAX_FREQ]

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    freq_bands = list(range(0, MAX_FREQ, FREQ_STEP))
    magFFT = np.abs(zc)
    phzFFT = np.angle(zc)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    for freq in freq_bands:
        data_row['FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)
        data_row['FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)
        data_row['FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)
        data_row['FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)

        data_row['FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_STEP])
        data_row['FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_STEP])
        data_row['FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_STEP])
        data_row['FFT_Mag_min%d' % freq] = np.min(magFFT[freq: freq + FREQ_STEP])

        data_row['FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_STEP])
        data_row['FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_STEP])
        data_row['FFT_Phz_max%d' % freq] = np.max(phzFFT[freq: freq + FREQ_STEP])
        data_row['FFT_Phz_min%d' % freq] = np.min(phzFFT[freq: freq + FREQ_STEP])

    data_row['FFT_Rmean'] = realFFT.mean()
    data_row['FFT_Rstd'] = realFFT.std()
    data_row['FFT_Rmax'] = realFFT.max()
    data_row['FFT_Rmin'] = realFFT.min()
    data_row['FFT_Imean'] = imagFFT.mean()
    data_row['FFT_Istd'] = imagFFT.std()
    data_row['FFT_Imax'] = imagFFT.max()
    data_row['FFT_Imin'] = imagFFT.min()

    data_row['FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    data_row['FFT_Rstd__first_6000'] = realFFT[:6000].std()
    data_row['FFT_Rmax_first_6000'] = realFFT[:6000].max()
    data_row['FFT_Rmin_first_6000'] = realFFT[:6000].min()
    data_row['FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    data_row['FFT_Rstd_first_18000'] = realFFT[:18000].std()
    data_row['FFT_Rmax_first_18000'] = realFFT[:18000].max()
    data_row['FFT_Rmin_first_18000'] = realFFT[:18000].min()

    del xcz
    del zc
    # gc.collect()

    sigs = [seg]
    for freq in range(0, MAX_FREQ + FREQ_STEP, FREQ_STEP):
        if freq == 0:
            xc_ = des_filter(seg, high=FREQ_STEP)
        elif freq == MAX_FREQ:
            xc_ = des_filter(seg, low=freq)
        else:
            xc_ = des_filter(seg, low=freq, high=freq + FREQ_STEP)
        sigs.append(pd.Series(xc_))

    for window in [50, 200, 1000]:
        roll_mean = seg.rolling(window).mean().dropna()
        roll_std = seg.rolling(window).std().dropna()
        sigs.append(pd.Series(roll_mean))
        sigs.append(pd.Series(roll_std))

    for span in [30, 300, 3000]:
        exp_mean = seg.ewm(span).mean().dropna()
        exp_std = seg.ewm(span).std().dropna()
        sigs.append(pd.Series(exp_mean))
        sigs.append(pd.Series(exp_std))

    for i, sig in enumerate(sigs):

        data_row['mean_%d' % i] = sig.mean()
        data_row['std_%d' % i] = sig.std()
        data_row['max_%d' % i] = sig.max()
        data_row['min_%d' % i] = sig.min()

        data_row['mean_change_abs_%d' % i] = np.mean(np.diff(sig))
        data_row['mean_change_rate_%d' % i] = np.mean(np.nonzero((np.diff(sig) / sig[:-1]))[0])
        data_row['abs_max_%d' % i] = np.abs(sig).max()
        data_row['abs_min_%d' % i] = np.abs(sig).min()

        data_row['std_first_50000_%d' % i] = sig[:50000].std()
        data_row['std_last_50000_%d' % i] = sig[-50000:].std()
        data_row['std_first_10000_%d' % i] = sig[:10000].std()
        data_row['std_last_10000_%d' % i] = sig[-10000:].std()

        data_row['avg_first_50000_%d' % i] = sig[:50000].mean()
        data_row['avg_last_50000_%d' % i] = sig[-50000:].mean()
        data_row['avg_first_10000_%d' % i] = sig[:10000].mean()
        data_row['avg_last_10000_%d' % i] = sig[-10000:].mean()

        data_row['min_first_50000_%d' % i] = sig[:50000].min()
        data_row['min_last_50000_%d' % i] = sig[-50000:].min()
        data_row['min_first_10000_%d' % i] = sig[:10000].min()
        data_row['min_last_10000_%d' % i] = sig[-10000:].min()

        data_row['max_first_50000_%d' % i] = sig[:50000].max()
        data_row['max_last_50000_%d' % i] = sig[-50000:].max()
        data_row['max_first_10000_%d' % i] = sig[:10000].max()
        data_row['max_last_10000_%d' % i] = sig[-10000:].max()

        data_row['max_to_min_%d' % i] = sig.max() / np.abs(sig.min())
        data_row['max_to_min_diff_%d' % i] = sig.max() - np.abs(sig.min())
        data_row['count_big_%d' % i] = len(sig[np.abs(sig) > 500])
        data_row['sum_%d' % i] = sig.sum()

        data_row['mean_change_rate_first_50000_%d' % i] = np.mean(
            np.nonzero((np.diff(sig[:50000]) / sig[:50000][:-1]))[0])
        data_row['mean_change_rate_last_50000_%d' % i] = np.mean(
            np.nonzero((np.diff(sig[-50000:]) / sig[-50000:][:-1]))[0])
        data_row['mean_change_rate_first_10000_%d' % i] = np.mean(
            np.nonzero((np.diff(sig[:10000]) / sig[:10000][:-1]))[0])
        data_row['mean_change_rate_last_10000_%d' % i] = np.mean(
            np.nonzero((np.diff(sig[-10000:]) / sig[-10000:][:-1]))[0])

        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            data_row['percentile_p{}_{}'.format(p, i)] = np.percentile(sig, p)
            data_row['abd_percentile_p{}_{}'.format(p, i)] = np.percentile(np.abs(sig), p)

        data_row['trend_%d' % i] = add_trend_feature(sig)
        data_row['abs_trend_%d' % i] = add_trend_feature(sig, abs_values=True)
        data_row['abs_mean_%d' % i] = np.abs(sig).mean()
        data_row['abs_std_%d' % i] = np.abs(sig).std()

        data_row['mad_%d' % i] = sig.mad()
        data_row['kurt_%d' % i] = sig.kurtosis()
        data_row['skew_%d' % i] = sig.skew()
        data_row['med_%d' % i] = sig.median()

        # data_row['Hilbert_mean_%d' % i] = np.abs(hilbert(sig)).mean()
        data_row['Hann_window50_%d' % i] = (convolve(sig, hann(50), mode='same') / sum(hann(50))).mean()
        data_row['Hann_window500_%d' % i] = (convolve(sig, hann(500), mode='same') / sum(hann(500))).mean()

        data_row['classic_sta_lta0_mean_%d' % i] = classic_sta_lta(sig, 50, 1000).mean()
        data_row['classic_sta_lta1_mean_%d' % i] = classic_sta_lta(sig, 500, 10000).mean()
        data_row['classic_sta_lta2_mean_%d' % i] = classic_sta_lta(sig, 5000, 100000).mean()
        data_row['classic_sta_lta3_mean_%d' % i] = classic_sta_lta(sig, 3333, 6666).mean()
        data_row['classic_sta_lta4_mean_%d' % i] = classic_sta_lta(sig, 10000, 25000).mean()

        no_of_std = 2
        for w in [10, 100, 500]:
            signal_mean = sig.rolling(window=w).mean()
            signal_std = sig.rolling(window=w).std()
            data_row['high_bound_mean_win{}_{}'.format(w, i)] = (signal_mean + no_of_std * signal_std).mean()
            data_row['low_bound_mean_win{}_{}'.format(w, i)] = (signal_mean - no_of_std * signal_std).mean()

        data_row['range_inf_4000_%d' % i] = feature_calculators.range_count(sig, -np.inf, -4000)
        data_row['range_4000_inf_%d' % i] = feature_calculators.range_count(sig, 4000, np.inf)
        for l, h in [[-4000, -2000], [-2000, 0], [0, 2000], [2000, 4000]]:
            data_row['range_{}_{}_{}'.format(np.abs(l), np.abs(h), i)] = feature_calculators.range_count(sig, l, h)

        data_row['iqr0_%d' % i] = np.subtract(*np.percentile(sig, [75, 25]))
        data_row['iqr1_%d' % i] = np.subtract(*np.percentile(sig, [95, 5]))
        data_row['ave10_%d' % i] = stats.trim_mean(sig, 0.1)
        data_row['num_cross_0_%d' % i] = feature_calculators.number_crossing_m(sig, 0)
        data_row['ratio_value_number_%d' % i] = feature_calculators.ratio_value_number_to_time_series_length(sig)
        # data_row['var_larger_than_std_dev_%d' % i] = feature_calculators.variance_larger_than_standard_deviation(sig)
        data_row['ratio_unique_values_%d' % i] = feature_calculators.ratio_value_number_to_time_series_length(sig)
        data_row['abs_energy_%d' % i] = feature_calculators.abs_energy(sig)
        data_row['abs_sum_of_changes_%d' % i] = feature_calculators.absolute_sum_of_changes(sig)
        data_row['count_above_mean_%d' % i] = feature_calculators.count_above_mean(sig)
        data_row['count_below_mean_%d' % i] = feature_calculators.count_below_mean(sig)
        data_row['mean_abs_change_%d' % i] = feature_calculators.mean_abs_change(sig)
        data_row['mean_change_%d' % i] = feature_calculators.mean_change(sig)
        data_row['first_loc_min_%d' % i] = feature_calculators.first_location_of_minimum(sig)
        data_row['first_loc_max_%d' % i] = feature_calculators.first_location_of_maximum(sig)
        data_row['last_loc_min_%d' % i] = feature_calculators.last_location_of_minimum(sig)
        data_row['last_loc_max_%d' % i] = feature_calculators.last_location_of_maximum(sig)
        data_row['long_strk_above_mean_%d' % i] = feature_calculators.longest_strike_above_mean(sig)
        data_row['long_strk_below_mean_%d' % i] = feature_calculators.longest_strike_below_mean(sig)
        # data_row['cid_ce_0_%d' % i] = feature_calculators.cid_ce(sig, 0)
        # data_row['cid_ce_1_%d' % i] = feature_calculators.cid_ce(sig, 1)

        for j in [10, 50, ]:
            data_row['peak_num_p{}_{}'.format(j, i)] = feature_calculators.number_peaks(sig, j)
        for j in [1, 10, 50, 100]:
            data_row['spkt_welch_density_coeff{}_{}'.format(j, i)] = \
            list(feature_calculators.spkt_welch_density(sig, [{'coeff': j}]))[0][1]
        for j in [5, 10, 100]:
            data_row['c3_c{}_{}'.format(j, i)] = feature_calculators.c3(sig, j)
        for j in [5, 10, 50, 100, 1000]:
            data_row['autocorrelation_auto{}_{}'.format(j, i)] = feature_calculators.autocorrelation(sig, j)
        for j in [10, 100, 1000]:
            data_row['time_rev_asym_stat_t{}_{}'.format(j, i)] = feature_calculators.time_reversal_asymmetry_statistic(
                sig, j)
        for j in range(1, 5):
            data_row['kstat_k{}_{}'.format(j, i)] = stats.kstat(sig, j)
            data_row['moment_m{}_{}'.format(j, i)] = stats.moment(sig, j)
        for j in range(1, 3):
            data_row['kstatvar_k{}_{}'.format(j, i)] = stats.kstatvar(sig, j)
        for j in [5, 10, 50, 100]:
            data_row['binned_entropy_b{}_{}'.format(j, i)] = feature_calculators.binned_entropy(sig, j)

    return data_row

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

    data = pd.DataFrame(columns=sorted(list(X[0].keys())))
    for x in X:
        data = data.append(x, ignore_index=True)
    label = np.asarray(Y)
    data['ttf'] = label

    del X
    del Y

    l.append((pid,data,load_index),)

def load_test_multiprocess(l,test_path,test_file,pid):
    X = []

    for f in tqdm(test_file,postfix=pid):
        seg = pd.read_csv(os.path.join(test_path, f), dtype={'acoustic_data': np.int16, })['acoustic_data']
        # if normalize:
        #     seg = (seg - acoustic_data_mean) / acoustic_data_std
        segment = create_features2(seg)
        X.append(segment)

    data = pd.DataFrame(columns=sorted(list(X[0].keys())))
    for x in X:
        data = data.append(x, ignore_index=True)
    seg_id = [tf.split('.')[0] for tf in test_file]
    data['seg_id'] = seg_id

    del X

    l.append((pid,data,))

if __name__ == '__main__':
    with open('../sample_end_indexs_{}.pkl'.format(num_samples),'rb') as f:
        sample_end_indexs = pickle.load(f)

    assert len(sample_end_indexs) == num_samples
    assert len(sample_end_indexs) == len(set(sample_end_indexs))

    # trainset_path = '/data1/jianglibin/earthquake/train.csv'
    dataset = pd.read_csv(train_csv_path, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    x = dataset['acoustic_data']
    y = dataset['time_to_failure']
    # if normalize:
    #     x = (dataset['acoustic_data'] - acoustic_data_mean) / acoustic_data_std
    print('CSV Data Loaded!')

    del dataset
    gc.collect()

    # load train data
    num_worker = 5
    assert num_samples % num_worker == 0
    num_each_worker = int(len(sample_end_indexs)/num_worker)
    end_index_each_worker = []
    for i in range(num_worker):
        if i < (num_worker-1):
            end_index_each_worker.append(sample_end_indexs[i*num_each_worker:(i+1)*num_each_worker])
        else:
            end_index_each_worker.append(sample_end_indexs[i*num_each_worker:])
    l = []
    with Manager() as manager:
        l_process = manager.list()
        workers = [Process(target=load_train_multiprocess, args=(l_process, x, y, end_index_each_worker[pid], pid)) for pid in range(num_worker)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        l = list(l_process)
        l = sorted(l,key=lambda x:x[0])

    train_data = []
    data_index= []
    for d in l:
        train_data.append(d[1])
        data_index.extend(d[2])
    train_data = pd.concat(train_data,ignore_index=True,sort=False)
    assert train_data.shape[0] == num_samples
    assert data_index == sample_end_indexs # data align check
    print('Train data Loaded!({} samples, shape:{})'.format(num_samples,train_data.shape))

    train_data.to_csv('/data2/jianglibin/earthquake/data/LGB/train_data_{}.csv'.format(train_data.shape))
    # train_data.to_csv('train_data_{}.csv'.format(train_data.shape))

    del x
    del y
    gc.collect()

    # load test data
    num_worker = 12
    with open('../test_file.pkl', 'rb') as f:
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
    for d in l:
        test_data.append(d[1])
    test_data = pd.concat(test_data, ignore_index=True,sort=False)
    assert test_data.shape[0] == test_num
    # data align check
    test_file = [tf.split('.')[0] for tf in test_file]
    assert test_data['seg_id'].values.tolist() == test_file
    print('Test Data Loaded!(2624 samples, shape:{})'.format(test_data.shape))

    test_data.to_csv('/data2/jianglibin/earthquake/data/LGB/test_data_{}.csv'.format(test_data.shape))
    # test_data.to_csv('test_data_{}.csv'.format(test_data.shape))
