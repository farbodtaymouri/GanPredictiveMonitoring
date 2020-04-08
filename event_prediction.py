
# Define models with the use of minibatch
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
import torch.nn.init  as init
import pandas as pd
import random
import pprint
from torch.nn.utils.rnn import pad_sequence
import pathlib
import os
import bottleneck as bn
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
device=torch.device('cuda:0')
plt.style.use('ggplot')


# Define an RNN model (The generator)
class LSTMGenerator(nn.Module):
    def __init__(self, seq_len, input_size, batch, hidden_size , num_layers, num_directions):
        super().__init__()
        self.input_size = input_size
        self.h = torch.randn(num_layers * num_directions ,batch, hidden_size)
        self.c = torch.randn(num_layers * num_directions ,batch, hidden_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.25, batch_first =True, bidirectional = False)


        latent_vector_size =50 * batch
        self.linear1 = nn.Linear(batch * seq_len *hidden_size, latent_vector_size)
        # self.linear2 = nn.Linear(latent_vector_size,batch*seq_len*hidden_size)
        self.linearHC = nn.Linear(num_layers *hidden_size *batch, latent_vector_size)
        # self.linearHCO = nn.Linear(3*latent_vector_size,batch*seq_len*hidden_size )
        self.linearHCO = nn.Linear( 3 *latent_vector_size ,batch *seq_len *input_size )




        # Define sigmoid activation and softmax output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x = x.view((1,x.size()[0], x.size()[1]))
        # Pass the input tensor through each of our operations
        # print("inputsize:", x.size())
        output, (h ,c) = self.lstm(x, (self.h, self.c))
        # print("inputsize:", x.size(),"output size:", output.size())
        # print("h size:", h.size(),"c size:", c.size())
        self.h = h.detach()
        self.c = c.detach()

        # Executing Fully connected network
        # print("The size of output:", output.size(), h.size(), c.size())
        u = output.reshape((output.size()[0 ] *output.size()[1 ] *output.size()[2]))
        u = self.relu(self.linear1(u))
        # print("The size of lninera1:", u.size())
        # u = self.linear2(u)

        # Flating h and feeding it into a linear layer
        uH = F.leaky_relu(self.linearHC(h.reshape((h.size()[0 ] *h.size()[1 ] *h.size()[2]))))
        uC = F.leaky_relu(self.linearHC(c.reshape((c.size()[0 ] *c.size()[1 ] *c.size()[2]))))
        uHCO = torch.cat((uH ,uC ,u))
        uHCO = self.linearHCO(uHCO)
        u= uHCO

        # output = u.view((output.size()[0],output.size()[1],output.size()[2]))
        output = u.view((output.size()[0],output.size()[1],self.input_size))
        # print("output size finally:", output.size())


        return output


####################################################################################################
# Define an RNN model (The discriminator)
class LSTMDiscriminator(nn.Module):
    def __init__(self, seq_len, input_size, batch, hidden_size, num_layers, num_directions):
        self.batch = batch
        super().__init__()
        self.h = torch.randn(num_layers * num_directions, batch, hidden_size)
        self.c = torch.randn(num_layers * num_directions, batch, hidden_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.25, batch_first=True, bidirectional=False)
        # h0 = torch.randn(,1, 513)
        # c0 = torch.randn(1,1, 513)

        latent_vector_size = 50 * batch
        self.linear1 = nn.Linear(batch * seq_len * hidden_size, latent_vector_size)
        self.linearHC = nn.Linear(num_layers * hidden_size * batch, latent_vector_size)
        # self.linearHCO = nn.Linear(3*latent_vector_size,batch*seq_len*input_size )
        self.linearHCO = nn.Linear(3 * latent_vector_size, batch * seq_len * input_size)
        self.linear2 = nn.Linear(batch * seq_len * input_size, 100)
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, batch)

        # h0.data *=0.001
        # c0.data *=0.001

        # Define sigmoid activation and softmax output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x = x.view((1,x.size()[0], x.size()[1]))
        # Pass the input tensor through each of our operations
        output, (h, c) = self.lstm(x, (self.h, self.c))
        # print("inputsize:", x.size(),"output size:", output.size())
        self.h = h.detach()
        self.c = c.detach()

        # Executing Fully connected network
        # print("The size of output:", output.size(), h.size(), c.size())
        u = output.reshape((output.size()[0] * output.size()[1] * output.size()[2]))
        u = self.relu(self.linear1(u))
        # u = self.linear2(u)

        # Flating h and feeding it into a linear layer
        uH = F.leaky_relu(self.linearHC(h.reshape((h.size()[0] * h.size()[1] * h.size()[2]))))
        uC = F.leaky_relu(self.linearHC(c.reshape((c.size()[0] * c.size()[1] * c.size()[2]))))
        uHCO = torch.cat((uH, uC, u))
        uHCO = self.linearHCO(uHCO)
        u = F.relu(self.linear2(uHCO))
        u = F.relu(self.linear3(u))
        u = self.linear4(u)

        # output = u.view((output.size()[0],output.size()[1],output.size()[2]))
        # output = u.view((output.size()[0],output.size()[1],input_size))
        output = u

        # Reshaping into (batch,-1)
        # tensor([[-0.1050],
        # [ 0.0327],
        # [-0.0260],
        # [-0.1059],
        # [-0.1055]], grad_fn=<ViewBackward>)
        output = output.reshape((self.batch, -1))

        return output

####################################################################################################
def one_hot_encoding(batch, no_events, y_truth):
    '''
    batch : the batch size
    no_events : the number of events
    y_truth : the ground truth labels

    example:
      tensor([[8.],
        [6.],
        [0.],
        [0.],
        [8.]])

    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])'''

    z = torch.zeros((batch, no_events))
    for i in range(z.size()[0]):
        z[i, y_truth[i].long()] = 1

    # print(z)
    return z.view(batch, 1, -1)

####################################################################################################
####################################################################################################
def model_eval_test(modelG, mode, obj):
    '''
    This module is for validation and testing the Generator
    @param modelG: Generator neural network
    @param mode: 'validation', 'test', 'test-validation'
    @param obj: A data object created from "Input" class that contains the required information
    @return: The accuracy of the Generator
    '''
    # set the evaluation mode (this mode is necessary if you train with batch, since in test the size of batch is different)
    rnnG = modelG
    rnnG.eval()



    validation_loader = obj.validation_loader
    test_loader = obj.test_loader
    batch = obj.batch
    #events = list(np.arange(0, len(obj.unique_event) + 1))
    events = list(np.arange(0, len(obj.unique_event) ))
    prefix_len = obj.prefix_len


    if (mode == 'validation'):
        data_loader = validation_loader
    elif (mode == "test"):
        data_loader = test_loader
    elif (mode == 'test-validation'):
        data_loader = test_loader + validation_loader


    predicted = []
    accuracy_record = []
    mistakes = {}

    accuracy_record_2most_probable = []
    y_truth_list = []
    y_pred_last_event_list = []

    for mini_batch in iter(data_loader):

        x = mini_batch[0];
        y_truth = mini_batch[1]
        # When we create mini batches, the length of the last one probably is less than the batch size, and it makes problem for the LSTM, therefore we skip it.
        if (x.size()[0] < batch):
            continue
        # print("x.size()", x.size())

        # Executing LSTM
        y_pred = rnnG(x[:, :, events])
        # print("y_pred:\n", y_pred, y_pred.size())

        # Just taking the last predicted element from each the batch
        y_pred_last = y_pred[0: batch, prefix_len - 1, :]
        y_pred_last = y_pred_last.view((batch, 1, -1))
        y_pred_last_event = torch.argmax(F.softmax(y_pred_last, dim=2), dim=2)

        y_truth_list += list(y_truth.flatten().data.cpu().numpy().astype(int))
        y_pred_last_event_list += list(y_pred_last_event.flatten().data.cpu().numpy().astype(int))

        y_pred_second_last = y_pred[0: batch, prefix_len - 2, :]
        y_pred_second_last = y_pred_second_last.view((batch, 1, -1))
        y_pred_second_last_event = torch.argmax(F.softmax(y_pred_second_last, dim=2), dim=2)

        # We iterate over the minibatch
        for i in range(x.size()[0]):

            if (y_pred_last_event[i] == y_truth[i].long()):
                # print("inside if:", y_pred, y_truth[i])
                correct_prediction = 1
            else:
                # print("inside else:", y_pred, y_truth[i])
                correct_prediction = 0

                # Collecting the mistakes
                k = str(y_truth[i].detach()) + ":" + str(y_pred_last_event[i].detach()) + str(
                    y_pred_second_last_event[i].detach())
                if (k not in mistakes):
                    mistakes[k] = 1
                else:
                    mistakes[k] += 1

            # Considering the second most probable
            if ((y_pred_second_last_event[i] == y_truth[i].long()) or (y_pred_last_event[i] == y_truth[i].long())):
                correct_prediction_2most_probable = 1
            else:
                correct_prediction_2most_probable = 0

            # accuracy_record.append(correct_prediction/float(len(y_pred)))
            accuracy_record.append(correct_prediction)
            accuracy_record_2most_probable.append(correct_prediction_2most_probable)
            predicted.append(y_pred)

    rnnG.train()
    # computing F1scores wiethed
    weighted_precision, weighted_recall, weighted_f1score, support = precision_recall_fscore_support(y_truth_list,
                                                                                            y_pred_last_event_list,
                                                                                            average='weighted',
                                                                                            labels=events)
    # computing F1score per each label
    precision, recall, f1score, support = precision_recall_fscore_support(y_truth_list, y_pred_last_event_list, average=None, labels=events)

    if (mode == 'test'):
        #pprint.pprint(mistakes)
        if(os.path.isfile(obj.path+'/results.txt')):
            with open(obj.path+'/results.txt', "a") as fout:
                pprint.pprint(mistakes, stream=fout)
        else:
            with open(obj.path+'/results.txt', "w") as fout:
                pprint.pprint(mistakes, stream=fout)

        with open(obj.path + '/results.txt', "a") as fout:
            fout.write("Turth: first prediction, second prediction\n" +
                       "total number of predictions:"+ str(len(accuracy_record))+','+str(np.sum(accuracy_record)) +
                       "\n The accuracy of the model with the most probable event:" + str(np.mean(accuracy_record))+
                       "\n The accuracy of the model with the 2 most probable events:" +str(np.mean(accuracy_record_2most_probable))+
                       '\n The list of activity names:' + str(events) +
                       '\n The precision per activity names:' + str(precision) +
                       '\n The recall per activity names:' + str(recall) +
                       '\n The F1 score per activity names:' + str(f1score) +
                       '\n The support per activity names:' + str(support) +
                       '\n The weighted precision, recall, and F1-score are: ' + str(weighted_precision) + ',' + str(weighted_recall) + ',' + str(weighted_f1score) + '\n')

        #fout.close()



    print("Labels:", events)
    print("Wighted Precision:", weighted_precision)
    print("Wighted Recall:", weighted_recall)
    print("Wighted F1score:", weighted_f1score)
    print("---------------------")
    print("Labels:", events)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1score:", f1score)
    print("Support:", support)
    print("Turth: first prediction, second prediction\n")
    print("total number of predictions:", len(accuracy_record), np.sum(accuracy_record))
    print("The accuracy of the model with the most probable event:", np.mean(accuracy_record))
    print("The accuracy of the model with the 2 most probable events:", np.mean(accuracy_record_2most_probable))
    return np.mean(accuracy_record)
####################################################################################################

def train(rnnG, rnnD, optimizerD, optimizerG, obj, epoch):
    '''
    @param rnnG: Generator neural network
    @param rnnD: Discriminator neural network
    @param optimizerD:  Optimizer of the discriminator
    @param optimizerG:  Optimizer of the generator
    @param obj:       A data object created from "Input" class that contains the training,test, and validation datasets and other required information
    @param epoch:    The number of epochs
    @return: Generator and Discriminator
    '''

    # Training Generator
    #epoch = 30
    #events = list(np.arange(0, len(obj.unique_event) + 1))
    events = list(np.arange(0, len(obj.unique_event) ))
    gen_loss_pred = []
    disc_loss_tot = []
    gen_loss_tot = []
    accuracy_best = 0

    for i in tqdm(range(epoch)):
        for mini_batch in iter(obj.train_loader):

            x = mini_batch[0];
            y_truth = mini_batch[1]
            # When we create mini batches, the length of the last one probably is less than the batch size, and it makes problem for the LSTM, therefore we skip it.
            if (x.size()[0] < obj.batch):
                continue
            # print(x[:,:,events], x.size(),'\n', y_truth)
            # -----------------------------------------------------------------------------------------------------

            # Training discriminator
            optimizerD.zero_grad()

            # Executing LSTM
            y_pred = rnnG(x[:, :, events])
            # print("y_pred:\n", y_pred, y_pred.size())

            # Just taking the last predicted element from each the batch
            y_pred_last = y_pred[0:obj.batch, obj.prefix_len - 1, :]
            y_pred_last = y_pred_last.view((obj.batch, 1, -1))
            # print("y_pred:", y_pred_last)

            # Converting the labels into one hot encoding
            y_truth_one_hot = one_hot_encoding(obj.batch, len(events), y_truth)

            # Creating synthetic and realistic datasets
            ##data_synthetic = torch.cat((x[:,:,events],F.softmax(y_pred_last,dim=2)), dim=1)
            y_pred_last_event = torch.argmax(F.softmax(y_pred_last, dim=2), dim=2)
            y_pred_one_hot = one_hot_encoding(obj.batch, len(events), y_pred_last_event)
            data_synthetic = torch.cat((x[:, :, events], y_pred_one_hot), dim=1)

            # Realistinc dataset
            data_realistic = torch.cat((x[:, :, events], y_truth_one_hot), dim=1)



            # Training Discriminator on realistic dataset
            discriminator_realistic_pred = rnnD(data_realistic)
            disc_loss_realistic = F.binary_cross_entropy(F.sigmoid(discriminator_realistic_pred),
                                                         torch.ones((obj.batch, 1)), reduction='sum')
            disc_loss_realistic.backward(retain_graph=True)

            # Training Discriminator on synthetic dataset
            discriminator_synthetic_pred = rnnD(data_synthetic)
            # print("disc pred:", discriminator_synthetic_pred)
            disc_loss_synthetic = F.binary_cross_entropy(F.sigmoid(discriminator_synthetic_pred),
                                                         torch.zeros((obj.batch, 1)), reduction='sum')
            disc_loss_synthetic.backward(retain_graph=True)

            disc_loss_tot.append(disc_loss_realistic.detach() + disc_loss_synthetic.detach())

            optimizerD.step()

            if len(disc_loss_tot) % 1000 == 0:
                print("iter =------------------------------ i :", i, len(disc_loss_tot), " the Disc error is:",
                      ", the avg is:", np.mean(disc_loss_tot))

            #-------------------------------------------------------------------------------------------------------------------------

            # Training teh Generator
            # Training the prediction for the generator

            optimizerG.zero_grad()

            # Computing the loss of prediction
            lstm_loss_pred = F.binary_cross_entropy(F.sigmoid(y_pred_last), y_truth_one_hot, reduction='sum')
            gen_loss_pred.append(lstm_loss_pred.detach())
            lstm_loss_pred.backward(retain_graph=True)

            # Fooling the discriminator by presenting the synthetic dataset and considering the labels as the real ones
            discriminator_synthetic_pred = rnnD(data_synthetic)
            # print("disc pred:", discriminator_synthetic_pred)
            gen_fool_dic_loss = F.binary_cross_entropy(F.sigmoid(discriminator_synthetic_pred), torch.ones((obj.batch, 1)),
                                                       reduction='sum')
            gen_fool_dic_loss.backward(retain_graph=True)

            gen_loss_tot.append(lstm_loss_pred.detach() + gen_fool_dic_loss.detach())

            optimizerG.step()

            if len(gen_loss_tot) % 1000 == 0:
                print("iter =------------------------------ i :", i, len(gen_loss_tot), " the Gen error is:",
                      ", the avg is:", np.mean(gen_loss_tot))

        # Applying validation after several epoches
        # Early stopping (checking for every 5 iterations)
        #path = os.getcwd() + "/" + obj.dataset_name + '/event_prediction' + '/prefix_' + str(obj.prefix_len)
        path = obj.path
        #obj.path=path
        if i % 5 == 0:
            rnnG.eval()
            accuracy = model_eval_test(rnnG, 'validation', obj)
            rnnG.train()
            if (accuracy > accuracy_best):
                print("The validation set accuracy is:", accuracy)
                accuracy_best = accuracy

                #Writing down the model
                if(os.path.isdir(path)):
                    torch.save(rnnG, path+"/rnnG(validation).m")
                    torch.save(rnnD, path+"/rnnD(validation).m")
                else:
                    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                    torch.save(rnnG, path + "/rnnG(validation).m")
                    torch.save(rnnD, path + "/rnnD(validation).m")

    #Saving the models after training
    torch.save(rnnG, path + "/rnnG.m")
    torch.save(rnnD, path + "/rnnD.m")

    #plot_loss(gen_loss_pred, "Prediction loss", obj)
    plot_loss(gen_loss_tot, "Generator loss total", obj)
    plot_loss(disc_loss_tot, "Discriminator loss total", obj)

#########################################################################################################

def plot_loss(data_list, title, obj):
    '''
    #Plotting the input data
    @param data_list: A list of error values or accuracy values
    @param obj:
    @param title: A description of the datalist
    @return:
    '''
    if(title == "Generator loss total" ):
        if(hasattr(obj, 'plot')):
            obj.plot+=1
        else:
            obj.plot=1
    


    #plt.figure()
    plt.plot(bn.move_mean(data_list, window=100, min_count=1), label = title)
    plt.title(title+ ' prefix =' + str(obj.prefix_len) + ',' + "batch = " + str(obj.batch))
    plt.legend()

    tt =str(datetime.now()).split('.')[0].split(':')
    strfile = obj.path+'/'+title+ ', prefix =' + str(obj.prefix_len) + ',' + "batch = " +str(obj.batch) + str(obj.plot)
    plt.savefig(strfile)

    if(title == "Discriminator loss total"):
        plt.close()





