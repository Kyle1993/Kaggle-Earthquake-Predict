import torch
from torch import nn
import torch.nn.functional as F
import sys

sys.path.append('../')

from DL.config import *
from DL import dl_utils

def load_model(model_name,input_size=config.feature_num,output_size=config.extract_feature_num,drop=True):
    if model_name == 'Earthquake_RNN':
        return Earthquake_RNN()
    if model_name == 'Earthquake_LSTM':
        return Earthquake_LSTM(input_size=input_size,output_size=output_size,drop=drop)
    if model_name == 'Earthquake_CNN':
        return Earthquake_CNN(input_size=input_size,output_size=output_size,drop=drop)
    if model_name == 'Earthquake_RNNCNN':
        return Earthquake_RNNCNN()

class Earthquake_LSTM(nn.Module):
    def __init__(self,input_size=config.feature_num,middle=256,output_size=config.extract_feature_num,drop=False):
        super(Earthquake_LSTM,self).__init__()
        self.fc1 = nn.Linear(input_size,middle)
        self.rnn = nn.LSTM(middle,middle,batch_first=True,num_layers=2,bidirectional=True)
        self.fc2 = nn.Linear(middle*3,output_size)
        self.fc3 = nn.Linear(output_size,1)

        self.drop = drop

        self.h0 = dl_utils.totensor(torch.zeros((middle,)))
        self.c0 = dl_utils.totensor(torch.zeros((middle,)))

    def forward(self, x):
        batch_szie = x.shape[0]

        h0 = self.h0.repeat(4,batch_szie,1)
        c0 = self.c0.repeat(4,batch_szie,1)

        x = F.relu(self.fc1(x))
        output,(hn,cn) = self.rnn(x,(h0,c0))
        output = output.mean(dim=1)
        hn = hn.mean(dim=0)
        x = torch.cat([output,hn],dim=1)

        if self.drop:
            x = F.dropout(x,p=0.2)

        dl_feature = self.fc2(x)
        x = F.relu(dl_feature)
        x = self.fc3(x)

        return x.squeeze(),dl_feature

class Earthquake_RNN(nn.Module):
    def __init__(self):
        super(Earthquake_RNN,self).__init__()
        self.fc1 = nn.Linear(config.feature_num,256)
        self.rnn = nn.RNN(256,256,batch_first=True)
        self.fc2 = nn.Linear(256,512)
        self.fc3 = nn.Linear(512,1)

        self.h0 = dl_utils.totensor(torch.zeros((256,)))

    def forward(self, x):
        batch_szie = x.shape[0]
        h0 = self.h0.repeat(1,batch_szie,1)

        x = F.relu(self.fc1(x))
        output,hn = self.rnn(x,h0)

        x = F.relu(self.fc2(hn))
        x = self.fc3(x)

        return x.squeeze()

class Earthquake_CNN(nn.Module):
    def __init__(self, windows_size=[3,5,7],input_size=config.feature_num,output_size=config.extract_feature_num,drop=True):
        super(Earthquake_CNN,self).__init__()
        self.windows_size = windows_size
        self.cnn1 = [nn.Conv2d(1, 256, (ws, input_size), padding=(int((ws - 1) / 2),0)) for ws in windows_size]
        self.cnn2 = [nn.Conv2d(len(windows_size), 128, (ws, 256), padding=(int((ws - 1) / 2),0)) for ws in windows_size]
        self.pool = [nn.MaxPool1d(kernel_size=config.seg_size[0]) for _ in range(3)]
        self.drop = drop

        self.fc1 = nn.Linear(128*len(windows_size),output_size)
        self.fc2 = nn.Linear(output_size,1)

    def cuda_(self,gpu):
        self.cuda(gpu)
        for m in self.cnn1+self.cnn2+self.pool:
            m.cuda(gpu)

    def forward(self, x):
        x = x.unsqueeze(1)
        xs = [F.relu6(cnn(x)).permute(0,3,2,1) for cnn in self.cnn1]
        x = torch.cat(xs,dim=1)

        xs = [self.pool[i](F.relu6(self.cnn2[i](x)).squeeze()) for i in range(len(self.windows_size))]
        x = torch.cat(xs,dim=1).squeeze()

        if self.drop:
            x = F.dropout(x,p=0.2)

        dl_feature = self.fc1(x)
        x = F.relu(dl_feature)
        x = self.fc2(x)

        return x.squeeze(),dl_feature


class Earthquake_RNNCNN(nn.Module):
    def __init__(self, windows_size=[3,5,7],feature_num=config.feature_num):
        super(Earthquake_RNNCNN,self).__init__()
        self.windows_size = windows_size
        self.cnn1 = [nn.Conv2d(1, 128, (ws, feature_num), padding=(int((ws - 1) / 2),0)) for ws in windows_size]
        self.cnn2 = [nn.Conv2d(len(windows_size), 128, (ws, 128), padding=(int((ws - 1) / 2),0)) for ws in windows_size]
        self.pool = [nn.MaxPool1d(kernel_size=config.seg_size[0]) for _ in range(3)]
        self.fc_cnn = nn.Linear(128*len(windows_size),256)


        self.rnn = nn.LSTM(feature_num,256,batch_first=True)
        self.fc_rnn = nn.Linear(256,256)

        self.h0 = dl_utils.totensor(torch.zeros((256,)))
        self.c0 = dl_utils.totensor(torch.zeros((256,)))

        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,1)

    def cuda_(self,gpu):
        self.cuda(gpu)
        for m in self.cnn1+self.cnn2+self.pool:
            m.cuda(gpu)

    def forward(self, x):
        # cnn
        cnn_x = x.unsqueeze(1)
        cnn_xs = [F.relu6(cnn(cnn_x)).permute(0,3,2,1) for cnn in self.cnn1]
        cnn_x = torch.cat(cnn_xs,dim=1)
        cnn_xs = [self.pool[i](F.relu6(self.cnn2[i](cnn_x)).squeeze()) for i in range(len(self.windows_size))]
        cnn_x = torch.cat(cnn_xs,dim=1).squeeze()

        cnn_x = F.relu6(self.fc_cnn(cnn_x))

        batch_szie = x.shape[0]
        h0 = self.h0.repeat(1,batch_szie,1)
        c0 = self.c0.repeat(1,batch_szie,1)
        output,(hn,cn) = self.rnn(x,(h0,c0))
        rnn_x = F.relu(self.fc_rnn(hn)).squeeze()

        x = torch.cat([cnn_x,rnn_x],dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x.squeeze()

class Earthquake_NN(nn.Module):
    def __init__(self,in_num,middle=1024,drop=True):
        super(Earthquake_NN,self).__init__()
        self.fc1 = nn.Linear(in_num,in_num*4)
        self.fc2 = nn.Linear(in_num*4,in_num*4)
        self.fc3 = nn.Linear(in_num*4,1)
        self.drop = drop

    def forward(self, x):
        if self.drop:
            x = F.dropout(x,p=0.8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze()



if __name__ == '__main__':
    model = Earthquake_CNN()
    print(model.__class__.__name__)

    x = torch.randn((16,config.seg_size[0],config.feature_num))

    y,f = model(x)

    print(y.shape,f.shape)



