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
from data_processing import Data_set, NRC_Data_set
from graph import Graph
from sgcn_lstm import *
from sklearn.metrics import mean_squared_error, mean_absolute_error


torch.manual_seed(443)
torch.cuda.manual_seed(443)


# Create the parser
my_parser = argparse.ArgumentParser(description='List of argument')

# Add the arguments
my_parser.add_argument("-m", type=str, default="description of model learning")
my_parser.add_argument('--ex_train', type=str, default='/home/neuronS1/NRC/NRC_rehab_train_ND_02.csv',
                       help='the name of exercise.')
my_parser.add_argument('--ex_test', type=str, default='/home/neuronS1/NRC/NRC_rehab_train_ND_01.csv',
                       help='the name of exercise.')
my_parser.add_argument("--workdir", type=str, default="/home/neuronS1/NRC/my_rehab/caseRecord")

my_parser.add_argument('--epoch', type=int, default= 50,
                       help='number of epochs to train.')
my_parser.add_argument('--batch_size', type=int, default= 32,
                       help='training batch size.')
my_parser.add_argument('--lr', type=int, default= 0.0001,
                       help='initial learning rate for optimizer.')



args = my_parser.parse_args()

expNUM=str(len(os.listdir(args.workdir)))
workdir=os.path.join(args.workdir, 'exp'+expNUM)
if not os.path.exists(workdir):
    os.mkdir(workdir)

f = open(os.path.join(workdir, "description.txt"), 'w')
for i in vars(args).items():
    print("{:<15}: {}".format(i[0], i[1]), file=f)

"""Performance matric"""
# test 및 평가 함수~ 차후 구현 필요
def mean_absolute_percentage_error(y_pred, y_true): 
    #print("get patient score|| [model prediction] / [ground truth] : {}/{}".format(y_pred, y_true)
    return torch.mean(abs((y_true - y_pred) / torch.clip(y_true, 0.0001))) * 100


def train(model, trainLoader, optimizer, criterion):
    corr = 0
    running_loss = 0
    accuracy = 0
    rate = []

    for heat_data, labels in trainLoader:
        optimizer.zero_grad()
        inputs = heat_data.type(torch.cuda.FloatTensor).to(device)
        label = labels.type(torch.float32).to(device)
        
        prediction = model(inputs)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # print(loss.item(), len(trainLoader))
        for p, l in zip(prediction, label):
            rate.append(abs(p-l).item())
                # if rate < 0.003:
                    #     corr +=1
                #corr /= batch_size

    accuracy = max(rate)
    return running_loss/len(trainLoader), accuracy

def validation(model, testLoader, criterion):
    corr = 0
    running_loss = 0
    accuracy = 0
    rate = []

    print(" || testing Batches...")
    
    for heat_data, labels in testLoader:
        inputs = heat_data.type(torch.cuda.FloatTensor).to(device)
        label = labels.type(torch.float32).to(device)
        with torch.no_grad():
            prediction = model(inputs)
            loss = criterion(prediction, label)
            
        running_loss += loss.item()
        for p, l in zip(prediction, label):
            rate.append(abs(p-l).item())
          # if rate < 0.003:
            #     corr +=1
        #corr /= batch_size

    accuracy = max(rate)
    return running_loss/len(testLoader), accuracy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size
learning_rate = args.lr
beta1 = 0.9
beta2= 0.999


"""import the whole dataset"""
train_file= args.ex_train
trainDataset = NRC_Data_set(train_file)  # folder name -> Train.csv, Test.csv
graph = Graph(len(trainDataset.body_part))
trainLoader = DataLoader(trainDataset, batch_size=batch_size)

test_file= args.ex_test
testDataset = NRC_Data_set(test_file)  # folder name -> Train.csv, Test.csv
testLoader = DataLoader(testDataset, batch_size=16)


model = train_network(bias_mat_1=graph.bias_mat_1.to(device), bias_mat_2=graph.bias_mat_2.to(device)).to(device)
criterion = nn.HuberLoss(reduction='sum', delta=0.1)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), learning_rate, (beta1, beta2), weight_decay=1e-3)

loss_list = []
val_loss_list = []
acc_list = []
val_acc_list = []
min_loss=100000000000



print('\n NOw start training.... allocate device: ', device)
for e in range(args.epoch):
    print("EPOCH ", e)
    model.train()
    
    run_loss, run_acc = train(model, trainLoader, optimizer, criterion)

    print("LOSS: {0:0.2f} ERROR: {1:0.2f}".format(run_loss, run_acc))
    loss_list.append(run_loss)
    acc_list.append(run_acc)

    if e % 5 == 0:
        model.eval()
        run_loss, run_acc = validation(model, testLoader, criterion)
        print("[VALIDATION] LOSS: {0:0.2f} ERROR: {1:0.2f}".format(run_loss, run_acc))
        val_loss_list.append(run_loss)
        val_acc_list.append(run_acc)

        if min_loss > run_loss:
            min_loss = run_loss
            torch.save(model.state_dict(), workdir+'/rehab_best.pth')
            print("THE BEST MODEL IS UPDATED\n") 

        now = datetime.now()
        log ={}
        log['loss']=loss_list
        log['error']=acc_list
        log['val_error']=val_acc_list
        log['val_loss']=val_loss_list
        with open(os.path.join(workdir,'log_'+now.strftime('%Y-%m-%d %H:%M:%S')+"_"+str(e)+'.pkl'), 'wb') as f:
                pkl.dump(log, f)

    gc.collect()



# y_pred = t.sc2.inverse_transform(y_pred)#scaling 값을 원본 스케일로 복원
# test_y = data_loader.sc2.inverse_transform(test_y) 

# """Performance matric"""
# # test 및 평가 함수~ 차후 구현 필요
# def mean_absolute_percentage_error(y_pred, y_true): 
#     #print("get patient score|| [model prediction] / [ground truth] : {}/{}".format(y_pred, y_true))
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  
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
