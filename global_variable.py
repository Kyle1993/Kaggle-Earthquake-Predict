earthquake_point = [5656573,
                    50085877,
                    104677355,
                    138772452,
                    187641819,
                    218652629,
                    245829584,
                    307838916,
                    338276286,
                    375377847,
                    419368879,
                    461811622,
                    495800224,
                    528777114,
                    585568143,
                    621985672,
                    629145480-1,]

train_shape = (629145480, 2)
test_length = 150000
test_num = 2624
num_samples = 10000 #4150
intact_wave_num = 15
cv = 5
normalize = True
break_wave_rate = (earthquake_point[-1]-earthquake_point[-2]+earthquake_point[0])/train_shape[0]

acoustic_data_mean = 4.519467573700124
acoustic_data_std = 10.735707249510964

acoustic_abs_data_mean = 5.5474668
acoustic_abd_data_std = 10.242381

time_to_failure_data_mean = 5.678285
time_to_failure_data_std = 3.6726966

train_csv_path = '/data2/jianglibin/earthquake/train.csv'
test_file_path = '/data2/jianglibin/earthquake/test'

# train_csv_path = '/data1/jianglibin/earthquake/train.csv'
# test_file_path = '/data1/jianglibin/earthquake/test'