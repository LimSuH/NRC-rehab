import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from torch.utils.data import Dataset

import RAST_process as ppc

class Data_set(Dataset):
    def __init__(self, dir, scale:str='Y scailing',
                  cmd={'ADL': ['BIA', 'UNI', 'BIS'], 'ROM': ['BIA', 'UNI', 'BIS']}, kpNum='NRC_25', ceiling=288):
        self.num_repitation = 5
        self.num_channel = 3
        self.dir = dir
        self.kpNum = kpNum
        self.ppc = ppc.FrameProcessWrapper('custom', 'repeat',  ceiling=ceiling)
        self.sc1 = MinMaxScaler()
        self.sc2 = MinMaxScaler()
        self.cmd = cmd       
        self.scale=scale
        self.body_part, self.cate = self.body_parts()

         #choose valid label list
        self.categorial = []
        for k in list(cmd.keys()):
            for ci in self.cate[k]:
                self.categorial += self.cate[k][ci]

        self.num_joints = len(self.body_part) 
        self.dataset = []
        self.get_Data = []
        self.num_timestep = 100
        self.new_label = []
        self.train_x, self.train_y= self.import_dataset()
        print("\nwhole data num:",self.__len__())
        self.batch_size = self.train_y.shape[0]
                
    def body_parts(self):
        if self.dir.find('KIMORE') == -1: #none KIMORE
            if self.kpNum == 'NRC_25':
                body_parts = [0, 1, 2, 5, 6, 7, 8, 11 ,12, 
                       91, 95, 99, 100, 103, 104, 107, 111, 
                       112, 116, 120, 121, 124, 125, 128, 132]

            elif self.kpNum == 'NRC_27':
                body_parts=[0, 5, 6, 7, 8, 9, 10,
                93, 95, 96, 99, 100, 103, 104, 107, 108, 111,
                114, 116, 117, 120, 121, 124, 125, 128, 129, 132]

            else:
                body_parts = [i for i in range(133)]
            self.bf_kpnum = 133
            self.kpNum = len(body_parts)

            cate = {'ADL':{'BIA':[6,7,8], 'BIS':[9, 10], 'UNI': [1,2,3,4,5]},
                            'ROM':{'BIA':[11,12,13,14], 'BIS':[15], 'UNI': []}}
            
            self.slicing=(-6,-4)
        else: ## KIMORE
            print("This is KIMORE Dataset, right?")
            self.bf_kpnum = 25
            body_parts = range(25)
            cate = {'All':{'All':[1,2,3,4,5]}, 'separate':{'none':[]}}
            self.slicing=(-5, -4)
        return body_parts, cate
    
    def import_dataset(self):
        self.heat_npy = pd.read_csv(self.dir, usecols=['skeleton_path', 'label']).values.tolist()
        self.heat_npy.sort()

        train_x = []
        train_y = []
        for i, a in enumerate(self.heat_npy):
            if int(a[0][self.slicing[0]:self.slicing[1]]) not in self.categorial:
                print("\nno target data: ", a[0])
                continue
            try:
                data = self.ppc.doPreProc(np.load(a[0]).astype(np.float32))
            except:
                print("\nfail to load: ", a[0])
                continue
            data = data.reshape(data.shape[0], self.bf_kpnum, -1)
            proc_x = data[:, self.body_part, :3]

            if 'x' in self.scale.lower():
                proc_x= self.preprocessing_X(proc_x)
            train_y.append(a[1])
            train_x.append(proc_x)
            self.get_Data.append(a)
            print('\r Load DATA {} / {}'.format(i, len(self.heat_npy)), end='')

        return train_x, self.preprocessing_Y(train_y)
            
    def preprocessing_X(self, X_train):        
        X_train = X_train.reshape(-1, self.num_joints)
        X_train = self.sc1.fit_transform(X_train)         

        X_train = X_train.reshape(-1, self.num_joints, 3)
        return X_train
    
    def preprocessing_Y(self, Y_train):
        y_train = np.reshape(Y_train, (-1,1))
        if 'y' in self.scale.lower():
            y_train = self.sc2.fit_transform(y_train)
            print("\n there is ", self.scale, " Scaling.")

        return y_train

    def __len__(self):
        assert len(self.train_x) == len(self.train_y)
        return len(self.train_y) 
    
    def __getitem__(self, index):
        data = np.transpose(self.train_x[index], (2, 0, 1))
        data = np.expand_dims(data, axis=-1)
        label = self.train_y[index]

        return data, label
