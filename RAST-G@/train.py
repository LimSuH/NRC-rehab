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

random_seed = 443

torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


from data_processing import Data_set
from net import st_gcn




# Create the parser
my_parser = argparse.ArgumentParser(description='List of argument')

# Add the arguments
my_parser.add_argument("-m", type=str, default="description of model learning")
my_parser.add_argument('--ex_train', type=str, default='../Data/NRC_rehab_GT_NDtrain.csv',
                       help='the name of exercise.')
my_parser.add_argument('--ex_test', type=str, default='../Data/NRC_rehab_GT_NDtest.csv',
                       help='the name of exercise.')
my_parser.add_argument("--workdir", type=str, default="caseRecord")
my_parser.add_argument("--skeleton", type=str, default="NRC_25")
my_parser.add_argument("--scailing", type=str, default="Y")
my_parser.add_argument("--category_cmd", type=dict, 
                       default=
                       #{'All':['All']})
                        {'ADL': ['BIA', 'UNI', 'BIS'], 'ROM': ['BIA', 'UNI', 'BIS']})
my_parser.add_argument('--epoch', type=int, default= 200,
                       help='number of epochs to train.')
my_parser.add_argument('--batch_size', type=int, default=64,
                       help='training batch size.')
my_parser.add_argument('--lr', type=float, default= 0.003,
                       help='initial learning rate for optimizer.')
my_parser.add_argument('--num_heads', type=int, default=8,
                       help='TA wrapper heads num')
my_parser.add_argument('--input_frame', type=int, default=288,
                       help='Uniform input frame.')


args = my_parser.parse_args()
expNUM=str(max([int(i[3:]) for i in os.listdir(args.workdir)])+1)
# expNUM ='debug'
workdir=os.path.join(args.workdir, 'exp'+expNUM)
if not os.path.exists(workdir):
    os.mkdir(workdir)

f = open(os.path.join(workdir, "description.txt"), 'w')
for i in vars(args).items():
    print("{:<15}: {}".format(i[0], i[1]), file=f)

wandb.login()
run = wandb.init(project="RAST-G@", name=expNUM)
print("{:<15}: {}".format("log URL:", run.url), file=f)


"""Performance matric"""
def mean_absolute_percentage_error(y_pred, y_true): 
    return torch.mean(abs((torch.tensor(y_true) - torch.tensor(y_pred)) / torch.clip(y_true))).item()


def train(model, trainLoader, optimizer, criterion):
    corr = 0
    accuracy = 0
    running_loss = []
    rate = []
    bdx = 1

    for heat_data, labels in trainLoader:
        optimizer.zero_grad()
       
        inputs = heat_data.type(torch.FloatTensor).to(device)
        label = labels.type(torch.float32).to(device)

        prediction = model(inputs)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())
        bdx+=1
        for p, l in zip(prediction, label):
            rate.append(abs(p-l).item())
            corr +=1

    accuracy =sum(rate)/corr
    return sum(running_loss)/len(trainLoader), accuracy

def validation(model, testLoader, criterion):
    corr = 0
    accuracy = 0
    rate = []
    running_loss = []

    print(" || testing Batches...")
    
    for heat_data, labels in testLoader:
        inputs = heat_data.type(torch.FloatTensor).to(device)
        label = labels.type(torch.float32).to(device) # float32
        with torch.no_grad():
            prediction = model(inputs)
            loss = criterion(prediction, label)
            
        running_loss.append(loss.item())
    
        for p, l in zip(prediction, label):
            rate.append(abs(p-l).item())
            corr +=1
   
    accuracy =sum(rate)/corr
    return sum(running_loss)/len(testLoader), accuracy, rate


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size
learning_rate = args.lr
beta1 = 0.9
beta2= 0.999

"""import the whole dataset"""
trainDataset = Data_set(args.ex_train, 
                        scale=args.scailing, 
                        cmd=args.category_cmd, 
                        kpNum= args.skeleton, 
                        ceiling=args.input_frame)  # folder name -> Train.csv, Test.csv

trainLoader = DataLoader(trainDataset, batch_size=batch_size)

testDataset = Data_set(args.ex_test, 
                       scale=args.scailing, 
                       cmd=args.category_cmd, 
                       kpNum= args.skeleton, 
                       ceiling=args.input_frame)  # folder name -> Train.csv, Test.csv

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
        print(f"{name} : {value}", file=f)
    
    return model

graph_args={'layout':args.skeleton, 'strategy':'spatial'}
model = call_with_log(st_gcn.Model, 3, 256, graph_args, False, output_dim=1)
criterion = nn.HuberLoss(reduction='mean', delta=0.1)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), learning_rate, (beta1, beta2), weight_decay=1e-3)

loss_list = []
val_loss_list = []
acc_list = []
val_acc_list = []
val_gap_list = []
min_loss=1000
val_run_loss=0
val_run_acc=0

for i in inspect.getsource(optimizer.__class__).split('\n'):
    print("optimizer info: {}".format(i), file=f)
    break
for i in inspect.getsource(criterion.__class__).split('\n'):
    print("loss info: {}".format(i), file=f)
    print("\tdelta: ", criterion.delta, file =f)
    break

print('\n NOw start training.... allocate device: {}, record dir: {}'.format(device, expNUM))
for e in range(args.epoch):
    print("EPOCH ", e)
    model.train()
    
    run_loss, run_acc = train(model, trainLoader, optimizer, criterion)

    print("LOSS: {0:0.4f} ERROR: {1:0.2f}".format(run_loss, run_acc))
    loss_list.append(run_loss)
    acc_list.append(run_acc)

    if e % 5 == 0:
        model.eval()
        val_run_loss, val_run_acc, val_run_gap = validation(model, testLoader, criterion)
        print("[VALIDATION] LOSS: {0:0.4f} ERROR: {1:0.2f}".format(val_run_loss, val_run_acc))
        val_loss_list.append(val_run_loss)
        val_acc_list.append(val_run_acc)
        val_gap_list.append(val_run_gap)
        torch.save(model.state_dict(), workdir+'/rehab_best.pth')
        print("THE BEST MODEL IS UPDATED\n") 
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
        log['val_gap'] = val_gap_list
        with open(os.path.join(workdir,'log_'+now.strftime('%Y-%m-%d %H:%M:%S')+"_"+str(e)+'.pkl'), 'wb') as f:
                pkl.dump(log, f)
        
    run.log({'loss': run_loss, 'validation_loss': val_run_loss, 'error':run_acc, 'validation_error':val_run_acc})


    gc.collect()
run.finish()
f.close()