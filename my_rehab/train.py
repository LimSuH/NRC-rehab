import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch import Tensor 
import math 
import numpy as np

import argparse
import pickle as pkl
import numpy as np
import os
import gc
from datetime import datetime

random_seed = 443  # for reproducibility
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
# from IPython.core.debugger import set_trace
#import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from data_processing import Data_set
from graph import Graph
from sgcn_lstm import *
from sklearn.metrics import mean_squared_error, mean_absolute_error


torch.manual_seed(443)
torch.cuda.manual_seed(443)


# Create the parser
my_parser = argparse.ArgumentParser(description='List of argument')

# Add the arguments
my_parser.add_argument('--ex', type=str, default='..',
                       help='the name of exercise.', required=True)
my_parser.add_argument("--workdir", type=str, default="/home/neuronS1/NRC/my_rehab/caseRecord")
my_parser.add_argument('--lr', type=int, default= 0.0001,
                       help='initial learning rate for optimizer.')

my_parser.add_argument('--epoch', type=int, default= 10,
                       help='number of epochs to train.')

my_parser.add_argument('--batch_size', type=int, default= 1,
                       help='training batch size.')
my_parser.add_argument("-m", type=str, default="")
#my_parser.add_argument('Path',
#                       type=str,
#                       help='the path to list')

# Execute the parse_args() method
args = my_parser.parse_args()

expNUM=str(len(os.listdir(args.workdir)))
workdir=os.path.join(args.workdir, 'exp'+expNUM)
if not os.path.exists(workdir):
    os.mkdir(workdir)

f = open(os.path.join(workdir, "description.txt"), 'w')
f.write(args.m)

def train(model, trainLoader, optimizer, criterion):
    corr = 0
    running_loss = 0
    accuracy = 0
    index=0

    for heat_data, labels in trainLoader:
        optimizer.zero_grad()
        inputs = heat_data.type(torch.cuda.FloatTensor).to(device)
        label = labels.type(torch.LongTensor).to(device)
        
        prediction = model(inputs)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        prediction = prediction.argmax(1)
        # print(prediction.shape, label.shape)

        for p, l in zip(prediction, label):
            index+=1
            if p == l:
                corr +=1
        #corr /= batch_size

    accuracy = corr / index * 100
    return running_loss/len(trainLoader), accuracy

def validation(model, testLoader, criterion):
    corr = 0
    running_loss = 0
    accuracy = 0
    index=0

    print(" || testing Batches...")
    
    for heat_data, labels in testLoader:
        inputs = heat_data.type(torch.cuda.FloatTensor).to(device)
        label = labels.type(torch.LongTensor).to(device)
        with torch.no_grad():
            prediction = model(inputs)
            loss = criterion(prediction, label)
            
        running_loss += loss.item()
        prediction = prediction.argmax(1)
            # print(prediction.shape, label.shape)

        for p, l in zip(prediction, label):
            if p == l:
                corr +=1
            index+=1

    accuracy = corr / index * 100
    # print('\n')
    return running_loss/len(testLoader), accuracy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 32
learning_rate = 0.0001


"""import the whole dataset"""
train_file, test_file = args.ex
trainDataset = Data_set(train_file)  # folder name -> Train.csv, Test.csv
graph = Graph(len(trainDataset.body_part))
trainLoader = DataLoader(trainDataset, batch_size=batch_size)

testDataset = Data_set(test_file)  # folder name -> Train.csv, Test.csv
graph = Graph(len(testDataset.body_part))
testLoader = DataLoader(testDataset, batch_size=10)


model = traincell(AD=graph.AD, AD2=graph.AD2, bias_mat_1=graph.bias_mat_1, bias_mat_2=graph.bias_mat_2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(learning_rate)

loss_list = []
val_loss_list = []
acc_list = []
val_acc_list = []

print('\n NOw start training....')
for e in range(args.epoch):
    print("EPOCH ", e)
    model.train()
    
    run_loss, run_acc, conf_mat = train(model, trainLoader, optimizer, criterion)

    print("LOSS: {0:0.2f} ACCURACY: {1:0.2f}".format(run_loss, run_acc))
    loss_list.append(run_loss)
    acc_list.append(run_acc)

    if e % 2 == 0:
        model.eval()
        run_loss, run_acc, val_conf_mat = validation(model, testLoader, criterion)
        print("[VALIDATION] LOSS: {0:0.2f} ACCURACY: {1:0.2f}".format(run_loss, run_acc))
        val_loss_list.append(run_loss)
        val_acc_list.append(run_acc)

        if min_loss > run_loss:
            min_loss = run_loss
            torch.save(model.state_dict(), workdir+'/rehab_best.pth')
            print("THE BEST MODEL IS UPDATED\n") 

            now = datetime.now()
            log ={}
            log['loss']=loss_list
            log['acc']=acc_list
            log['conf_mat'] = conf_mat
            log['val_acc']=val_acc_list
            log['val_loss']=val_loss_list
            log['val_conf_mat'] = val_conf_mat
            with open(os.path.join(workdir,'log_'+now.strftime('%Y-%m-%d %H:%M:%S')+"_"+str(e)+'.pkl'), 'wb') as f:
                    pkl.dump(log, f)

    gc.collect()



# y_pred = t.sc2.inverse_transform(y_pred)#scaling 값을 원본 스케일로 복원
# test_y = data_loader.sc2.inverse_transform(test_y) 

"""Performance matric"""
# test 및 평가 함수~ 차후 구현 필요
def mean_absolute_percentage_error(y_true, y_pred): 
    print("get patient score|| [model prediction] / [ground truth] : {}/{}".format(y_pred, y_true))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  
# test_dev = abs(test_y-y_pred)
# mean_abs_dev = np.mean(test_dev)
# mae = mean_absolute_error(test_y, y_pred)
# rms_dev = sqrt(mean_squared_error(y_pred, test_y))
# mse = mean_squared_error(test_y,y_pred) 
# mape = mean_absolute_percentage_error(test_y, y_pred)
# print('Mean absolute deviation:', mae)
# print('RMS deviation:', rms_dev)
# print('MSE:', mse)
# print('MAPE: ', mape)
