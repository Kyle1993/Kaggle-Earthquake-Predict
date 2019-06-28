import numpy as np
import torch
import sys

sys.path.append('../')
from DL.config import *

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()


def totensor(data,):
    if isinstance(data, float):
        tensor = torch.FloatTensor([data])
    if isinstance(data, int):
        tensor = torch.LongTensor([data])
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).float()
    if isinstance(data, torch.Tensor):
        tensor = data.float()
    if config.gpu>=0:
        tensor = tensor.cuda(config.gpu)
    return tensor