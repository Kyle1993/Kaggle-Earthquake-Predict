from sklearn.model_selection import KFold
import random
import pickle
import os
from global_variable import *
from utils import *

def generate_sample_end_index_(num=num_samples):
    sample_step = int((earthquake_point[-1]-test_length)/num)
    end_indexs = []
    for i in range(num):
        end_index = i*sample_step + test_length + 1
        while cross_earthquake(end_index):
            end_index = random.randint(test_length+1,earthquake_point[-1])
        end_indexs.append(end_index)

    return end_indexs

sample_end_indexs = generate_sample_end_index_(num_samples)
assert len(sample_end_indexs) == num_samples
assert len(sample_end_indexs) == len(set(sample_end_indexs))


folds = KFold(n_splits=cv, shuffle=True, random_state=2019)

kfold = []
for f in folds.split(list(range(num_samples))):
    kfold.append(f)

# with open('sample_end_indexs_{}.pkl'.format(num_samples),'wb') as f:
#     pickle.dump(sample_end_indexs,f)
#
# with open('kfold_{}.pkl'.format(num_samples),'wb') as f:
#     pickle.dump(kfold,f)
#
# test_file_path = '/data2/jianglibin/earthquake/test'
# test_file = list(filter(lambda x:'seg_'in x,os.listdir(test_file_path)))
# assert len(test_file) == test_num
# with open('test_file.pkl','wb') as f:
#     pickle.dump(test_file,f)