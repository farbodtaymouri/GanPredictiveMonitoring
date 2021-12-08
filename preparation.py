
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import multiprocessing

class Input:
    #Class variables (remember that they are different than instance variables, and all instances or objects have access to them)
    path = '' #The location where the results will be written
    mode = ''   #Type of prediction task that the object will be used for, i.e., "event_prediction", "timestamp_prediction", "event_timestamp_prediction"
    dataset_name = '' #Name of the input dataset
    prefix_len = ''  #It is a number that shows the length of the considered prefixes
    batch = ''       #It is a number that shows size of used batch
    design_matrix = ''  # A matrix that stores the designed matrix (each activity is shown by one hot vector)
    design_matrix_padded = '' #A design matrix that is padded after creating the prefixes
    y = '' #The ground truth labels related to the "design_matrix_padded"
    unique_event = ''  #The list of unique events, including end of trace as "0"
    selected_columns = '' # List of considered columns, including event and other information
    timestamp_loc = ''    # The column index for timestamp feature
    train_inds=''     #Index of training instances
    test_inds=''      #Index of test instances
    validation_inds=''     #Index of validation instances
    train_loader = ''
    test_loader = ''
    validation_loader = ''




    #class methods can be called without creating objects (they have cls instead of self)
    #start from here
    @classmethod
    def run(cls,path, prefix, batch_size, mode = "event_prediction"):
        '''
        This method is the starting point for preparing an object to be used later in different prediction tasks.

        @param path: The location of the event log
        @param prefix: Size of the prefix
        @param batch_size: Size of batch
        @param mode: "event_prediction", "timestamp_prediction", "event_timestamp_prediction"
        @return:
        '''

        cls.prefix_len = prefix
        cls. batch = batch_size
        cls.dataset_name = path.split("/")[-1].split('.')[0]
        cls.mode = mode
        cls.path = os.getcwd() + "/" + cls.dataset_name + '/'+mode + '/prefix_' + str(cls.prefix_len)

        #Reading a file
        if(path.split('.')[-1] == 'csv'):  #path.split('.')[1]---> path.split('.')[-1] to handle path like ./dataset/filename.csv
            #Reading a CSV file
            #data_augment = cls.__read_csv(path)
            data_augment = cls.__read_csv_massive(path)
        elif(path.split('.')[-1] == 'pkl'):   # path.split('.')[1]---> path.split('.')[-1] to handle path like ./dataset/filename.csv
            data_augment = pickle.load(open(path, "rb"))
            print("The head of augmented with remaining and duration times:\n", data_augment.head(10))

        print(data_augment.shape, flush=True)
        #Creating a design matrix that shows one hot vector representation for activity IDs
        cls.design_matrix = cls.__design_matrix_creation(data_augment)

        #Creating prefix
        cls.prefix_creating(prefix, mode)

        #Determining the train,test, and validation sets
        cls.train_valid_test_index()

        # #Correcting test data
        # cls.testData_correction()

        #Creating minibatch
        cls.mini_batch_creation(batch_size)



    #################################################################################
    #Reading the CSV file
    @classmethod
    def __read_csv(cls, path):
        '''
        The input CSV is a file where the events is encoded into numerical activity IDs
        '''
        # Reaging files from CSV
        dat = pd.read_csv(path)
        print("Types:", dat.dtypes)
        # changing the data type from integer to category
        dat['ActivityID'] = dat['ActivityID'].astype('category')
        dat['CompleteTimestamp'] = dat['CompleteTimestamp'].astype('datetime64[ns]')
        print("Types after:", dat.dtypes)

        print("columns:", dat.columns)
        dat_group = dat.groupby('CaseID')
        print("Original data:", dat.head())
        print("Group by data:", dat_group.head())


        # Data Preparation
        # Iterating over groups in Pandas dataframe
        data_augment = pd.DataFrame()
        dat_group = dat.groupby('CaseID')

        total_iter = len(dat_group.ngroup())
        pbar = tqdm(total=total_iter)
        for name, gr in dat_group:
            # sorting by time
            gr.sort_values(by=['CompleteTimestamp'])
            # print (gr)

            # computing the duration time in seconds by differecning x[t+1]-x[t]
            #duration_time = gr.loc[:, 'CompleteTimestamp'].diff() / np.timedelta64(1, 's')
            duration_time = gr.loc[:, 'CompleteTimestamp'].diff() / np.timedelta64(1, 'D')
            # Filling Nan with 0
            duration_time.iloc[0] = 0
            # print ("duration time:\n", duration_time)

            # computing the remaining time
            length = duration_time.shape[0]
            remaining_time = [np.sum(duration_time[i + 1:length]) for i in range(duration_time.shape[0])]
            # print("Time to finish:\n", remaining_time)

            gr['duration_time'] = duration_time
            gr['remaining_time'] = remaining_time

            data_augment = data_augment.append(gr)

            # print("gr after:\n", gr)

            # break
            pbar.update(1)
        pbar.close()

        #For big inputs, its necessary to pickle it
        name = path.split(".")[0].split("/")[-1]
        pickle.dump(data_augment, open(name+".pkl", "wb"))

        print("Dataset with indicating remaining and duration times:\n", data_augment.head(10))
        return data_augment
    ################################################################################
    # Reading the CSV file
    @classmethod
    def __read_csv_massive(cls, path):
        '''
        The input CSV is a file where the events is encoded into numerical activity IDs
        see link https://stackoverflow.com/questions/40357434/pandas-df-iterrows-parallelization
        '''
        # Reaging files from CSV
        dat = pd.read_csv(path)
        print("Types:", dat.dtypes)
        # changing the data type from integer to category
        dat['ActivityID'] = dat['ActivityID'].astype('category')
        dat['CompleteTimestamp'] = dat['CompleteTimestamp'].astype('datetime64[ns]')
        print("Types after:", dat.dtypes)

        print("columns:", dat.columns)
        dat_group = dat.groupby('CaseID')
        print("Original data:", dat.head())
        print("Group by data:", dat_group.head())

        # create as many processes as there are CPUs on your machine
        num_processes = multiprocessing.cpu_count()
        # calculate the chunk size as an integer
        chunk_size = int(dat.shape[0] / num_processes)

        # will work even if the length of the dataframe is not evenly divisible by num_processes
        #chunks = [dat.ix[dat.index[i:i + chunk_size]] for i in range(0, dat.shape[0], chunk_size)]
        chunks = [dat.iloc[dat.index[i:i + chunk_size]] for i in range(0, dat.shape[0], chunk_size)]




        # create our pool with `num_processes` processes
        pool = multiprocessing.Pool(processes=num_processes)
        # apply our function to each chunk in the list
        results = pool.map(cls.func, chunks)        #Results is a list of dataframe [df1, df2,...df10]
        results = pd.concat(results)
        #return results

        name = path.split(".")[0].split("/")[-1]
        pickle.dump(results, open(name + ".pkl", "wb"))

        return results   # add return results for read_csv_massive
    ######################################################################################
    @classmethod
    def func(cls,dat):
        # Data Preparation (used by read_csv_massive)
        # Iterating over groups in Pandas dataframe
        data_augment = pd.DataFrame()
        dat_group = dat.groupby('CaseID')

        total_iter = len(dat_group.ngroup())
        pbar = tqdm(total=total_iter)
        for name, gr in dat_group:
            # sorting by time
            gr.sort_values(by=['CompleteTimestamp'])
            # print (gr)

            # computing the duration time in seconds by differecning x[t+1]-x[t]
            # duration_time = gr.loc[:, 'CompleteTimestamp'].diff() / np.timedelta64(1, 's')
            duration_time = gr.loc[:, 'CompleteTimestamp'].diff() / np.timedelta64(1, 'D')
            # Filling Nan with 0
            duration_time.iloc[0] = 0
            # print ("duration time:\n", duration_time)

            # computing the remaining time
            length = duration_time.shape[0]
            remaining_time = [np.sum(duration_time[i + 1:length]) for i in range(duration_time.shape[0])]
            # print("Time to finish:\n", remaining_time)

            gr['duration_time'] = duration_time
            gr['remaining_time'] = remaining_time

            data_augment = data_augment.append(gr)

            # print("gr after:\n", gr)

            # break
            pbar.update(1)
        pbar.close()


        #print("Dataset with indicating remaining and duration times:\n", data_augment.head(10))
        return data_augment
    #######################################################################################
    #Creating a design matrix (one hot vector representation)
    @classmethod
    def __design_matrix_creation(cls, data_augment):
        '''
        data_augment is pandas dataframe created after reading CSV input by "read_csv()" method
        '''
        # Creating a desing matrix (one hot vectors for activities), End of line (case) is denoted by class 0
        unique_event = sorted(data_augment['ActivityID'].unique())
        cls.unique_event = [0] + unique_event
        print("uniqe events:", unique_event)

        l = []
        for index, row in tqdm(data_augment.iterrows()):
            temp = dict()
            '''
            temp ={1: 0,
                  2: 0,
                  3: 1,
                  4: 0,
                  5: 0,
                  6: 0,
                  '0':0,
                  'duration_time': 0.0,
                  'remaining_time': 1032744.0}
            '''

            # Defning the columns we consider
            keys = ['0'] + list(unique_event) + ['duration_time', 'remaining_time']
            for k in keys:
                if (k == row['ActivityID']):
                    temp[k] = 1
                else:
                    temp[k] = 0

            temp['class'] = row['ActivityID']
            temp['duration_time'] = row['duration_time']
            temp['remaining_time'] = row['remaining_time']
            temp['CaseID'] = row['CaseID']

            l.append(temp)

        # Creating a dataframe for dictionary l
        design_matrix = pd.DataFrame(l)
        print("The design matrix is:\n", design_matrix.head(10))
        return design_matrix
    ################################################################################
    # Creating the desing matrix based on given prefix.
    @classmethod
    def prefix_creating(cls, prefix=2, mode = 'event_prediction'):


        if (mode == "timestamp_prediction"):
            clsN = cls.design_matrix.columns.get_loc('duration_time')
        elif (mode == "event_prediction"):
            clsN = cls.design_matrix.columns.get_loc('class')
        elif (mode == 'event_timestamp_prediction'):
            clsN = [cls.design_matrix.columns.get_loc('duration_time')] + [cls.design_matrix.columns.get_loc('class')]
            cls.timestamp_loc = cls.design_matrix.columns.get_loc('duration_time')
            cls.selected_columns = cls.unique_event + [cls.timestamp_loc]



        group = cls.design_matrix.groupby('CaseID')
        # Iterating over the groups to create tensors
        temp = []
        temp_shifted = []
        for name, gr in group:
            gr = gr.drop('CaseID', axis=1)
            # For each group, i.e., view, we create a new dataframe and reset the index
            gr = gr.copy(deep=True)
            gr = gr.reset_index(drop=True)

            # Commenting out this portion to prevent long tail samples for train/test
            # for example: from a series of events [3,5,6,8,6,0] no longer generats samples
            # like [8, 6, 0, 0], [6, 0, 0, 0] and [0, 0, 0, 0]
            #  
            # # adding a new row at the bottom of each case to denote the end of a case
            # new_row = [0] * gr.shape[1]
            # gr.loc[gr.shape[0]] = new_row
            # gr.iloc[gr.shape[0] - 1, gr.columns.get_loc('0')] = 1  # End of line is denoted by class 0

            # gr_shift = gr.shift(periods=-1, fill_value=0)
            # gr_shift.loc[gr.shape[0] - 1, '0'] = 1

            # Selecting only traces that has length greater than the defined prefix

            if (gr.shape[0] - 1 > prefix):
                for i in range(gr.shape[0]-prefix): # change range(gr.shape[0]) ---> range(gr.shape[0]-prefix)
                    # if (i+prefix == gr.shape[0]):
                    #   break
                    # print(gr.iloc[i:i+prefix])
                    temp.append(torch.tensor(gr.iloc[i:i + prefix].values, dtype=torch.float, requires_grad=False))


                    # Storing the next element after the prefix as the prediction class
                    try:
                        # print("the prediction:", "the i", i ,gr.iloc[i+prefix,cls])
                        temp_shifted.append(
                            torch.tensor([gr.iloc[i+prefix, clsN]], dtype=torch.float, requires_grad=False)) #change gr.iloc[i + prefix, clsN] ---> gr.iloc[-1, clsN]
                    except IndexError:
                        # Printing the end of sequence
                        # print("the prediction:", "ESLE the i", i ,0)
                        temp_shifted.append(torch.tensor([np.float16(0)], dtype=torch.float, requires_grad=False))
                    # print("****************************")


            # break
        desing_matrix_padded = pad_sequence(temp, batch_first=True)
        desing_matrix_shifted_padded = pad_sequence(temp_shifted, batch_first=True)

        # Saving the variables
        cls.design_matrix_padded = desing_matrix_padded
        cls.y = desing_matrix_shifted_padded
        #return desing_matrix_padded, desing_matrix_shifted_padded


        #Applying pad corrections
        cls.__pad_correction()

        print("The dimension of designed matrix:", cls.design_matrix_padded.size())
        print("The dim of ground truth:", cls.y.size())
        print("The prefix considered so far:", cls.design_matrix_padded.size()[1])


    ########################################################################################
    @classmethod
    def __pad_correction(cls):

        for i in range(cls.design_matrix_padded.size()[0]):
            u = (cls.design_matrix_padded[i, :, 0] == 1).nonzero()

            try:
                cls.design_matrix_padded[i, :, 0][u:] = 1
            except TypeError:
                pass
    ##################################################################################
    @classmethod
    def train_valid_test_index(cls):
        # Creating indexes by which obtaining train,test and validation sets


        train_inds = np.arange(0, round(cls.design_matrix_padded.size()[0] * .8))

        # Generating index for the test dataset
        test_inds = list(set(range(cls.design_matrix_padded.size()[0])).difference(set(train_inds)))
        validation_inds = test_inds[0:round(0.3 * len(test_inds))]
        test_inds = test_inds[round(0.3 * len(test_inds)):]


        cls.train_inds = train_inds
        cls.test_inds = test_inds
        cls.validation_inds = validation_inds
        print("Number of training instances:", len(train_inds))
        print("Number of testing instances:", len(test_inds))
        print("Number of validation instances:", len(validation_inds))

    #################################################################################
    @classmethod
    def testData_correction(cls):

        test_inds_new = []
        for i in cls.test_inds:
            # Checking how many stops are available in the prefix (we drop thoes prefixes with more than one stop element)
            # Remeber that the first column (index = 0) shows the end of sequence
            u = (cls.design_matrix_padded[i, :, 0] == 1).nonzero()
            if len(u) <= 1:
                test_inds_new.append(i)

        print("The number of test prefixes before correction:", len(cls.test_inds))
        print("The number of test prefixes after correction:", len(test_inds_new))

        cls.test_inds =  test_inds_new

    #################################################################################
    @classmethod
    def mini_batch_creation(cls, batch=4 ):
        train_data = TensorDataset(cls.design_matrix_padded[cls.train_inds], cls.y[cls.train_inds])
        train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)


        test_data = TensorDataset(cls.design_matrix_padded[cls.test_inds], cls.y[cls.test_inds])
        test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True)

        validation_data = TensorDataset(cls.design_matrix_padded[cls.validation_inds], cls.y[cls.validation_inds])
        validation_loader = DataLoader(dataset=validation_data, batch_size=batch, shuffle=True)




        cls.train_loader = train_loader
        cls.test_loader = test_loader
        cls.validation_loader = validation_loader
        print("The minibatch is created!")





