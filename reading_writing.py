
import numpy as np
import os
import xml.etree.ElementTree as ET
#import Variables as V
import xlsxwriter as XL
import re
import pandas as pd
from tqdm import tqdm
import pprint
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def Reading_log_all(log_path, destination_path):
    print("##########################################")
    # print" The actual body of gets started"

    # Reading log from "log.txt"
    total_trace = []
    # for k in range(500):
    # f=open("C:/Users/University/Desktop/Net and log/log_test/Real_log/BPIC15_2_log/BPIC15_2_log_%d.txt" %k)
    # f=open("C:/Users/University/Desktop/Net and log/log_test/Real_log/BPIC15_4_log/BPIC15_4_log_19.txt" ,"r")
    # f=open('C:/Users/University/Documents/chert/plg.txt','r')

    log_temp, case_name = Read_XES(log_path)
    # f=open(log_path, 'r')


    # This part only writes separate traces to the txt files
    '''for i in range(len(log_temp)):
        if not os.path.exists(destination_path + "/logs"):
            os.makedirs(destination_path + "/logs")
        # h=open(destination_path+"/logs"+"/trace_%i.txt" %i,"w")
        #h = open(destination_path + "/logs" + "/trace_%i.txt" % case_name[i], "w")
        h = open(destination_path + "/logs" + "/trace_"+ str(case_name[i])+ ".txt", "w")

        for j in range(len(log_temp[i])):
            h.write(log_temp[i][j] + " ")
        h.close()'''






    # reading all the lines, each line is
    '''temp=f.readlines()
    log_temp=[]
    for i in range(len(temp)):
        trace=temp[i].strip("\n").split(" ")
        log_temp.append(trace)

    temp=[]
    #print"The log before adding [T]:",log_temp
    f.close()'''

    # --------------------------
    #  this part is temporarili, remove it after use and use the above block
    # import checknevis as ch
    # log_temp=ch.Read_txt(log_path)
    # ---------------------------------------

    # log = []
    # for k in range(len(log_temp)):
    #     for i in range(len(log_temp[k])):
    #         # Just for working with realistic examples"
    #         # ----------
    #         if ("_" in log_temp[k][i]):
    #             log_temp[k][i] = log_temp[k][i].replace("_", "*")
    #         if (" " in log_temp[k][i]):
    #             log_temp[k][i] = log_temp[k][i].replace(" ", "#")
    #         # ----------

            #log_temp[k][i] = ("T_" + log_temp[k][i])
            ###log.append(log_temp[i])

    # finding uniqe traces
    total_trace = []
    dictionary_log = dict()

    k = 0
    for i in range(len(log_temp)):
        if (log_temp[i] not in total_trace):
            total_trace.append(log_temp[i])
            # This dictionary is only for tracking which traces are realted each uniqe trace
            # dictionary_log[k]=[i]
            # dictionary_log[k]=str(i)
            dictionary_log[k] = str(case_name[i])
            k += 1
        elif (log_temp[i] in total_trace):
            trace_index = total_trace.index(log_temp[i])
            # dictionary_log[trace_index].append(i)
            # dictionary_log[trace_index]=dictionary_log[trace_index]+','+str(i)
            dictionary_log[trace_index] = dictionary_log[trace_index] + ',' + str(case_name[i])


            # print "the log after adding [T]:",log

    print( "++++++++++++++++++++++++++++++++++++++++")

    # --------------------------------------------------------------
    ###--------------------Attention----------------------
    ###In some situations where we creat a model from the given log, using inductive miner in ProM, it is possible to have a model that
    # do not contain all the distinct events in the log, for example it is possible to have a model with transitions t1,t3,t5 which created
    # from this log=[[t1,t2,t5],[t1,t3,t5,t4],[t1,t5,t3]]. In order to avoid the future problem we remove thees type of events from the trace
    # when we want to pass the trace to the ILP, Also the Prom Does the same!
    # path="C:/Users/University/workspace/jav2/result/Place invariant/"+MODEL
    path = destination_path
    # trans, places, Incident_matrix,initial_place_marking=CCW.Preprocess()
    events_not_replayed = []

    '''for i in range(len(total_trace)):
        temp=[]
        #temp2=[]
        for j in range(len(total_trace[i])):
            if(total_trace[i][j] in trans):
                temp.append(total_trace[i][j])
            elif(total_trace[i][j] not in trans):
                events_not_replayed.append(total_trace[i][j])
        total_trace[i]=temp
        #events_not_replayed.append(temp2)
    if(len(events_not_replayed)!=0):
        #Wrintg not replayed events
        f=open(path+"/Events not replayed.txt","w")
        for i in range(len(events_not_replayed)):
                f.write(events_not_replayed[i]+" ")
        f.close()'''

    '''trans = V.transition
    places = V.p_name
    Incident_matrix = V.Inc_matrix
    initial_place_marking = V.initial_place_marking

    events_not_replayed = []

    for i in range(len(total_trace)):
        temp = []
        # temp2=[]
        for j in range(len(total_trace[i])):
            if (total_trace[i][j] in trans):
                temp.append(total_trace[i][j])
            elif (total_trace[i][j] not in trans):
                events_not_replayed.append(total_trace[i][j])
        total_trace[i] = temp
        # events_not_replayed.append(temp2)
    if (len(events_not_replayed) != 0):
        # Wrintg not replayed events
        events_not_replayed = list(set(events_not_replayed))
        f = open(path + "/Events not replayed.txt", "w")
        for i in range(len(events_not_replayed)):
            f.write(events_not_replayed[i] + " ")
        f.close()'''




    return total_trace, dictionary_log



    # ---------------------------------------------------------
    # print " total_trace:", total_trace[0]
    # return total_trace[0]
#############################################################################
#### Reading XES file

# For better understanding of how it works open a XES file(small one) with http://xmlgrid.net
# Remember that the XES file must be preprocessed such that only contain complete actions
def Read_XES(path):
    add = path
    tree = ET.parse(add)
    root = tree.getroot()

    # --New Added 30 August 2016-------------------------------------------
    # Some XML files have Namesapce and it is declered at the root like and represented by xmlns like: <pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">.
    # So in order to acces the tags inside the XML file, the name space must be mentioned!, The general form is like "{name_sapce}tag",
    # For example for reading places tags, the code is like below:
    #   for p in root.iter("{ http://www.pnml.org/version-2009/grammar/pnml }place"):
    #       print p
    # ------
    # First we need to extract the namespace, namely the value of xmlns
    # root.tag='{http://www.pnml.org/version-2009/grammar/pnml}pnml' but we need '{http://www.pnml.org/version-2009/grammar/pnml}'
    # For this issue we use regular expression library (re) of python
    m = re.match('\{.*\}', root.tag)
    # checking whether m is empty or no
    if (m):
        namespace = m.group(0)
    else:
        namespace = ''
        # ------------------------------------------------------------


    temp2=[]
    temp = []
    log = []
    case_name = []
    event_time_stamp=[]
    print ("We are before for")
    for t in root.iter(namespace +"trace"):
        # ----
        # Reading the case number, like "instance_294"
        for name in t.iter(namespace +"string"):
            # print name.attrib['key']
            if (name.attrib['key'] == 'concept:name'):
                try:
                    temp_name = name.attrib['value']
                    temp_name = temp_name.split("_")[1]
                    case_name.append(int(temp_name))
                except IndexError:
                    # When the case number is like A1001
                    temp_name = temp_name.split("_")[0]
                    case_name.append(temp_name)


                break
        # print "The caee_name:",case_name
        # ---------------------
        for e in t.iter(namespace +"event"):
            for r in e.iter(namespace +"string"):
                # print r.attrib
                if (r.attrib['key'] == 'concept:name'):
                    # print r.attrib['value']
                    temp.append(r.attrib['value'])

            # New 14/10/2019
            # Searcing to the time of events
            for r in e.iter(namespace + 'date'):
                if (r.attrib['key'] == "time:timestamp"):
                    temp2.append(r.attrib['value'])
        event_time_stamp.append(temp2)
        temp2 = []


                    # print "the temp is:", temp
        log.append(temp)
        temp = []





        # print t.attrib

    print ("We are after for")
    print ("The len log is:", len(log))
    print ("the len case_name:", len(case_name))
    ###-------
    # re-arranging the position of the log, starting from zero
    # Before arranging case_name[0]='297' and log[0]= case 297
    # After  arranging, case_name[0]='0' and log[0]=case 0
    # Example:

    #    >>> list1 = [3,2,4,1, 1]
    #    >>> list2 = ['three', 'two', 'four', 'one', 'one2']
    #    >>> list1, list2 = zip(*sorted(zip(list1, list2)))
    #    >>> list1
    #        (1, 1, 2, 3, 4)
    #    >>> list2
    #        ('one', 'one2', 'two', 'three', 'four')

    print ("The len log is:", log)
    print ("the len case_name:", case_name)
    ##case_name, log=zip(*sorted(zip(case_name,log)))

    # ---------------------------------------

    ##printting the mean length of the traces
    men = 0
    for i in range(len(log)):
        men += len(log[i])
    print ("the mean lenght of traces:", men / len(log))





    return log, case_name
    #return log, case_name, event_time_stamp

###########################################################################
#######################################################3
#Reading huge amount of XES files (more than 4GB) and extract only events and write them as a new XES file
def Xes_Read_Massive(path):

    f=open(path)
    log=dict()
    flag_event=0
    flag_trace=0
    trace_numebr = 0
    for line in f:

        line = line.strip()
        #print line
        if (line == "<trace>"):
            flag_trace=1
            temp=[]

        #Reading Case id
        if(flag_trace ==1 and flag_event ==0):
            if ('string key="concept:name"' in line):
                case_id = line.strip('<').strip('/>').split("value=")[1]
                case_id = case_id.strip('\"')
                case_id = case_id.replace(" ", "**")
                case_id = case_id.replace("_", "*#*")
                case_id = case_id + '_' + str(trace_numebr)
                log[case_id] = ''
                #print case_id
                trace_numebr+=1


        #truning on the flag by which reading events of a trace
        if(flag_trace == 1 and line == "<event>"):
            flag_event =1

        #Reading the events of a trace
        if (flag_trace == 1 and flag_event ==1):
            if ('string key="concept:name"' in line):
                    event=line.strip('<').strip('/>').split("value=")[1]
                    event = event.strip('\"')
                    event = event.replace(" ", "**")
                    event = event.replace("_", "#")
                    #event = event.replace("*+*", "*#*")
                    temp.append(event)

        #turning off the event flag
        if( line == "</event>"):
            flag_event =0


        #Turning off the trace flag
        if (line == "</trace>"):
            flag_trace =0
            log[case_id] =temp



    #Creating a new XES file
    XES_Create(log)
    return log
    #return log      # It is like {'97': ['A', 'C', 'E', 'H', 'G', 'F', 'D', 'B'],.....}



###################################################################################
##This is a temporal file for XES file creation, called from the above function
class Trace:
    # This class is only relating to the XES_Creation() method
    def __init__(self, moves, id_dic):
        self.moves = moves
        self.id_dic = id_dic


def XES_Create(log_path, name_of_xes='XES'):
    '''log_path is a dictionary of log. See the output of Xes_Read_Massive(path)'''

    ##To understand the code for creation the XES file, open one of them in the browser and then look at it!!


    #Creating object
    trace_obj = []
    log=log_path
    for key in log:
        trace_obj.append(Trace(log[key], [key]))






    # Creating XES file
    log = ET.Element("log")
    for t in trace_obj:
        #print "The id is:", t.id_dic

        # id_dic is like [25,848,927] or [10]
        for j in t.id_dic:
            trace = ET.SubElement(log, 'trace')

            ET.SubElement(trace, "string", key="concept:name", value= j)
            for e in t.moves:
                event = ET.SubElement(trace, 'event')
                ET.SubElement(event, "string", key="concept:name", value=e)
                ET.SubElement(event, "string", key="lifecycle:transition", value="complete")

    '''s=['a','b','c']
    for e in s:
        event=ET.SubElement(trace,'event')
        ET.SubElement(event,"string",key="concept:name", value=e)
        ET.SubElement(event,"string", key="lifecycle:transition", value="complete")'''

    tree = ET.ElementTree(log)

    tree.write(os.getcwd() + '/' + 'EventOnly.xes.xml')
    #tree.write(os.getcwd()+'/'+'EventOnly.xes.xml', xml_declaration=True, encoding='utf-8', method="xml")
    print ("The creation is done!")

##################################################################################
def read_csv_raw(path):
    '''
    This method reads a CSV file (the one that is exported from ProM). Create numerical activity IDs for events such that it can be used in Pytorch
    '''
    dat = pd.read_csv(path, index_col= False)

    #Correcting the formats
    dat['completeTime'] = dat['completeTime'].astype('datetime64[ns]')
    dat['startTime'] = dat['startTime'].astype('datetime64[ns]')
    dat['event'] = dat['event'].astype('category')

    #Grouping based caseID and then sort based on 'completeTime'
    dat.sort_values(['completeTime']).groupby('case')

    print("the data is:\n",dat.head(20))
    print("Types:\n", dat.dtypes)

    #Mapping events into numerical activity IDs
    unique_event = sorted(dat['event'].unique())
    print("The unique events are:\n", len(unique_event), unique_event)

    #Repalcing events with numerical ID
    dat_numID = dat.replace(unique_event, np.arange(1,len(unique_event)+1) )

    print("the data numerical Act ID is:\n", dat_numID.head(20))



    #Writing to file:
    #Selecting the columns that we want
    columns = ['case', 'event', 'startTime', 'completeTime']
    #Creating a new header for columns
    header = ['CaseID', 'ActivityID', 'startTime','CompleteTimestamp']

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    # writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')
    #
    # dat_numID.to_excel( writer, index = False, columns=columns, header = header)
    # writer.save()

    dat_numID.to_csv("pandas.csv", index = False, columns=columns, header = header )
    # f = open("demofile2.csv", "w")
    # f.write(str(dat_numID))

#########################################################################################
#Reporing the mean, max, min length of traces as well as other information
def log_statistics(path):
    '''

    @param path: A CSV file (usually after applying "read_csv_raw()"
    @return:
    '''
    dat = pd.read_csv(path, index_col=False)
    dat_group = dat.groupby('CaseID')

    events=[]
    activity_num=0
    trace_max=0
    trace_min=1000000
    trace_avg=0
    trace_num=0
    for name, gr in dat_group:
        if len(gr) > trace_max:
            trace_max = len(gr)
        if len(gr) < trace_min:
            trace_min = len(gr)
        trace_num +=1

    print("The dataset is:", path.split(".")[0].split("/")[-1])
    print("The maximum length of trace is:", trace_max)
    print("The minimum length of trace is:", trace_min)
    print("The avg length of trace is:", len(dat['ActivityID'])/float(trace_num))
    print("The number of events:", len(dat['ActivityID']))
    print("The number of activities:", len(dat['ActivityID'].unique()))
    print("The number of traces:", trace_num)

########################################################################################


