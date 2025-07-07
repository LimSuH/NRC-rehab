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
import inspect
import wandb

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
from net import st_gcn
from sklearn.metrics import mean_squared_error, mean_absolute_error


torch.manual_seed(443)
torch.cuda.manual_seed(443)


# Create the parser
my_parser = argparse.ArgumentParser(description='List of argument')

# Add the arguments
my_parser.add_argument("-m", type=str, default="description of model learning")
my_parser.add_argument('--ex_train', type=str, default='/home/neuronS1/NRC/forTrueSTGCN_train.csv',
                       help='the name of exercise.')
my_parser.add_argument('--ex_test', type=str, default='/home/neuronS1/NRC/forTrueSTGCN_test.csv',
                       help='the name of exercise.')
my_parser.add_argument("--workdir", type=str, default="/home/neuronS1/NRC/my_rehab/caseRecord")
my_parser.add_argument("--category_cmd", type=dict, default={'ADL':['BIA', 'UNI', 'BIS'], 'ROM':['BIA', 'UNI', 'BIS']})

my_parser.add_argument('--epoch', type=int, default= 200,
                       help='number of epochs to train.')
my_parser.add_argument('--batch_size', type=int, default= 64,
                       help='training batch size.')
my_parser.add_argument('--lr', type=float, default= 0.005,
                       help='initial learning rate for optimizer.')


args = my_parser.parse_args()

expNUM=str(len(os.listdir(args.workdir)))
workdir=os.path.join(args.workdir, 'exp'+expNUM)
if not os.path.exists(workdir):
    os.mkdir(workdir)

f = open(os.path.join(workdir, "description.txt"), 'w')
for i in vars(args).items():
    print("{:<15}: {}".format(i[0], i[1]), file=f)

wandb.login()
run = wandb.init(project="NRC-rehabilitation", name=expNUM)
print("{:<15}: {}".format("log URL:", run.url), file=f)


"""Performance matric"""
# test 및 평가 함수~ 차후 구현 필요
def mean_absolute_percentage_error(y_pred, y_true): 
    #print("get patient score|| [model prediction] / [ground truth] : {}/{}".format(y_pred, y_true)
    return torch.mean(abs((torch.tensor(y_true) - torch.tensor(y_pred)) / torch.clip(y_true))).item()


def train(model, trainLoader, optimizer, criterion):
    corr = 0
    #running_loss = 0
    accuracy = 0
    rate = []
    gt = []
    pred = []
    running_loss = []

    for heat_data, labels in trainLoader:
        optimizer.zero_grad()
        inputs = heat_data.type(torch.cuda.FloatTensor).to(device)
        label = labels.type(torch.float32).to(device)
        
        prediction = model(inputs)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())

        # print(loss.item(), len(trainLoader))
        # gt.append(label)
        # pred.append(prediction)
        for p, l in zip(prediction, label):
            rate.append(abs(p-l).item())
            corr +=1
            # print(abs(p-l).shape, p[0], l[0])
                # if rate < 0.003:
                #         corr +=1
                # corr /= batch_size

    # accuracy = max(rate)
    accuracy =sum(rate)/corr
    return sum(running_loss)/len(trainLoader), accuracy

def validation(model, testLoader, criterion):
    corr = 0
    #running_loss = 0
    accuracy = 0
    rate = []
    gt = []
    pred = []
    running_loss = []

    print(" || testing Batches...")
    
    for heat_data, labels in testLoader:
        inputs = heat_data.type(torch.cuda.FloatTensor).to(device)
        label = labels.type(torch.float32).to(device)
        with torch.no_grad():
            prediction = model(inputs)
            loss = criterion(prediction, label)
            
        running_loss.append(loss.item())
        # gt.append(label.item())
        # pred.append(prediction.item())
        for p, l in zip(prediction, label):
            rate.append(abs(p-l).item())
            corr +=1
            # print(abs(p-l).shape, p[0], l[0])
                # if rate < 0.003:
                #         corr +=1
                # corr /= batch_size

    # accuracy = max(rate)
    accuracy =sum(rate)/corr
    return sum(running_loss)/len(testLoader), accuracy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size
learning_rate = args.lr
beta1 = 0.9
beta2= 0.999


"""import the whole dataset"""
train_file= args.ex_train
trainDataset = Data_set(train_file, cmd=args.category_cmd)  # folder name -> Train.csv, Test.csv
graph = Graph(len(trainDataset.body_part))
trainLoader = DataLoader(trainDataset, batch_size=batch_size)

test_file= args.ex_test
testDataset = Data_set(test_file, cmd=args.category_cmd)  # folder name -> Train.csv, Test.csv
testLoader = DataLoader(testDataset, batch_size=16)

def call_with_log(func, *args, **kwargs):
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    
    model = func(*args, **kwargs).to(device)
    print("\nmodel info:", file=f)
    print("[STGCN]\n",model.st_gcn_networks[0], file=f)
    print("[Temporal Attention]\n",model.temp_att, file=f)
    for name, value in bound.arguments.items():
        print(f"{name} = {value}", file=f)
    
    return model

#model = call_with_log(train_network(bias_mat_1=graph.bias_mat_1.to(device), bias_mat_2=graph.bias_mat_2.to(device)))
graph_args={'layout':'NRC_25', 'strategy':'spatial'}
model = call_with_log(st_gcn.Model, 3, 256, graph_args, False)
criterion = nn.HuberLoss(reduction='mean', delta=0.8)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), learning_rate, (beta1, beta2), weight_decay=1e-3)

loss_list = []
val_loss_list = []
acc_list = []
val_acc_list = []
min_loss=100000000000
val_run_loss=0
val_run_acc=0

for i in inspect.getsource(optimizer.__class__).split('\n'):
    print("optimizer info: {}".format(i), file=f)
    break

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
        val_run_loss, val_run_acc = validation(model, testLoader, criterion)
        print("[VALIDATION] LOSS: {0:0.2f} ERROR: {1:0.2f}".format(val_run_loss, val_run_acc))
        val_loss_list.append(val_run_loss)
        val_acc_list.append(val_run_acc)

        if min_loss > run_loss:
            min_loss = run_loss
            torch.save(model.state_dict(), workdir+'/rehab_best.pth')
            print("THE BEST MODEL IS UPDATED\n") 

        now = datetime.now()
        log ={}
        log['loss']=loss_list
        log['error']=acc_list

        log['val_loss']=val_loss_list
        log['val_error']=val_acc_list
        with open(os.path.join(workdir,'log_'+now.strftime('%Y-%m-%d %H:%M:%S')+"_"+str(e)+'.pkl'), 'wb') as f:
                pkl.dump(log, f)
        
    run.log({'loss': run_loss, 'validation_loss': val_run_loss, 'error':run_acc, 'validation_error':val_run_acc})

    gc.collect()
run.finish()



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

f.close()