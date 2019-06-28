import sys
sys.path.append('../')
from global_variable import *

class Config():

    def __init__(self):
        self.lr = 1e-4
        self.lr_decay = 0.1
        # self.lr_decay_step = 5000
        self.batch_size = 64
        self.validate_batch_size = 96
        self.validate_step = 50
        self.seg_size = (25,6000)


        self.use_adam = True

        self.feature_num = 203
        self.extract_feature_num = 10

        self.gpu = 0

        self.epoch = 10
        self.epoch_decay = 4

        self.ks_check = False
        self.threshold_d = 0.05
        self.threshold_p = 0
    def to_dict(self):
        return vars(self)

    def to_str(self):
        config_str = '\n'
        for k,v in sorted(vars(self).items(),key = lambda x:x[0] ):
            config_str += '{}: {}\n'.format(k,v)

        return config_str

config = Config()

if __name__ == '__main__':
    print(config.to_str())