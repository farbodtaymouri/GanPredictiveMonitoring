from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import torch.nn as nn
from scipy.special import softmax
import torchvision
from torch.autograd import Variable
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=1000)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import random
import pprint
from torch.nn.utils.rnn import pad_sequence
import preparation as pr
import event_prediction as ep
import event_timestamp_prediction as etp
device=torch.device('cuda:0')
plt.style.use('ggplot')


def run(path, mode, prefix=4, epoch=1):
    '''

    @param path: The path to the CSV file
    @param mode:  One of "event_prediction" or 'event_timestamp_prediction'
    @param prefix: Size of the prefix
    @param epoch:  Number of epochs
    @return:
    '''

    if(mode == "event_prediction"):
        obj = pr.Input()
        #Preparing the input
        obj.run(path = path , prefix= prefix, batch_size= 5, mode=mode)

        #Initializing a generator
        selected_columns = obj.unique_event
        print("Selected columns:", selected_columns)
        rnnG = ep.LSTMGenerator(seq_len = obj.prefix_len, input_size = len(selected_columns), batch = obj.batch, hidden_size= 2*len(selected_columns) , num_layers = 2, num_directions = 1)
        optimizerG = torch.optim.Adam(rnnG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        #Initializing a discriminator
        rnnD = ep.LSTMDiscriminator(seq_len = obj.prefix_len+1, input_size = len(selected_columns), batch = obj.batch, hidden_size = 2*len(selected_columns), num_layers =2, num_directions = 1)
        optimizerD = torch.optim.Adam(rnnD.parameters(), lr=0.0002, betas=(0.5, 0.999))


        #Training and testing
        ep.train(rnnD=rnnD, rnnG=rnnG,optimizerD=optimizerD, optimizerG = optimizerG, obj=obj, epoch=epoch)
        #Loading the model from the validation
        rnng_validation = torch.load(obj.path+"/rnnG(validation).m")
        for i in range(5):
            ep.model_eval_test(modelG= rnnG, mode='test', obj = obj)
            ep.model_eval_test(modelG= rnng_validation, mode='test', obj = obj)
            obj.train_valid_test_index()
            obj.mini_batch_creation(batch=obj.batch)


    elif(mode == 'event_timestamp_prediction'):
        obj = pr.Input()
        # Preparing the input
        obj.run(path=path, prefix=prefix, batch_size=5, mode=mode)

        #Initializing a generator
        rnnG = etp.LSTMGenerator(seq_len = obj.prefix_len, input_size = len(obj.selected_columns), batch = obj.batch, hidden_size= 2*len(obj.selected_columns) , num_layers = 2, num_directions = 1)
        optimizerG = torch.optim.Adam(rnnG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Initializing a discriminator
        rnnD = etp.LSTMDiscriminator(seq_len=obj.prefix_len + 1, input_size=len(obj.selected_columns), batch=obj.batch, hidden_size=2 * len(obj.selected_columns), num_layers=2, num_directions=1)
        optimizerD = torch.optim.Adam(rnnD.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Training and testing
        etp.train(rnnD=rnnD, rnnG=rnnG, optimizerD=optimizerD, optimizerG=optimizerG, obj=obj, epoch=epoch)
        # Loading the model from the validation
        rnng_validation = torch.load(obj.path + "/rnnG(validation).m")
        for i in range(3):
            etp.model_eval_test(modelG=rnnG, mode='test', obj=obj)
            etp.model_eval_test(modelG=rnng_validation, mode='test', obj=obj)
            obj.train_valid_test_index()
            obj.mini_batch_creation(batch=obj.batch)




    return obj



