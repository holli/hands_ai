import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import glob
import sys
import pathlib
import time
import random
import json
import math
import pickle
import pdb
from collections import OrderedDict, Iterable, defaultdict
import re

import torch
#from torch import *
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor, tensor
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
# from torchvision import transforms, utils

import fastai
import fastai.vision
#import fastai.basic_data
#from fastai import vision
#from fastai.vision.image import *

_ = torch.manual_seed(42); np.random.seed(42); random.seed(42)



# import sklearn
# from sklearn import preprocessing
# sklearn forces some deprecation warnings, this is one way to avoid seeing them
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",category=DeprecationWarning)
#     def _warn_ignore(*args, **kwargs):
#         pass
#     _warn_original = warnings.warn
#     warnings.warn = _warn_ignore
#     import sklearn
#     warnings.warn = _warn_original

