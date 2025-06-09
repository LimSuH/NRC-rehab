import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal
from IPython.core.debugger import set_trace
from torch.utils.data import Dataset

import my_preprocess as ppc

index_Spine_Base=0
index_Spine_Mid=4
index_Neck=8
index_Head=12   # no orientation
index_Shoulder_Left=16
index_Elbow_Left=20
index_Wrist_Left=24
index_Hand_Left=28
index_Shoulder_Right=32
index_Elbow_Right=36
index_Wrist_Right=40
index_Hand_Right=44
index_Hip_Left=48
index_Knee_Left=52
index_Ankle_Left=56
index_Foot_Left=60  # no orientation    
index_Hip_Right=64
index_Knee_Right=68
index_Ankle_Right=72
index_Foot_Right=76   # no orientation
index_Spine_Shoulder=80
index_Tip_Left=84     # no orientation
index_Thumb_Left=88   # no orientation
index_Tip_Right=92    # no orientation
index_Thumb_Right=96  # no orientation

class Data_set():
    def __init__(self, dir, scale:str='Y scailing'):
        self.num_repitation = 5
        self.num_channel = 3
        self.dir = dir
        self.ppc = ppc.FrameProcessWrapper('max', 'repeat')
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        # self.dir_label = dir_label
        self.scale=scale
        self.body_part = self.body_parts()       
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 100
        self.new_label = []
        self.train_x, self.train_y= self.import_dataset()
        self.batch_size = self.train_y.shape[0]
        self.num_joints = len(self.body_part)
        # self.scaled_x, self.scaled_y = self.preprocessing()
                
    def body_parts(self):
        if self.dir.find('KIMORE') == -1:
            body_parts = [0, 1, 2, 5, 6, 7, 8, 11 ,12, 
                       91, 95, 99, 100, 103, 104, 107, 111, 
                       112, 116, 120, 121, 124, 125, 128, 132]
            #[0, 5, 6, 7, 8, 9, 10,
            #     93, 95, 96, 99, 100, 103, 104, 107, 108, 111,
            #     114, 116, 117, 120, 121, 124, 125, 128, 129, 132]
     
        else:
            body_parts = [index_Spine_Base, index_Spine_Mid, index_Neck, index_Head, index_Shoulder_Left, 
                        index_Elbow_Left, index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, 
                        index_Wrist_Right, index_Hand_Right, index_Hip_Left, 
                        
                        
                        
                        
                        index_Knee_Left, index_Ankle_Left, 
                        index_Foot_Left, index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right, index_Spine_Shoulder, 
                        index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
                        ]

        return body_parts
    
    def import_dataset(self):
        heat_npy = pd.read_csv(self.dir).values.tolist()
        # heat_npy.sort()

        train_x = []
        train_y = []
        print("\n there is ", self.scale, " Scaling.")
        for i, a in enumerate(heat_npy):
            data = self.ppc.doPreProc(np.load(a[0]).astype(np.float32))
            data = data.reshape(data.shape[0], 133, -1)
            proc_x = data[:, self.body_part, :3]

            # if data.shape[1] * data.shape[2] == 532:
            #     s2 = np.expand_dims(data[:, self.body_part, 0:-1:4], axis=-1)
            #     s3 = np.expand_dims(data[:, self.body_part, 1:-1:4], axis=-1)
            #     s4 = np.expand_dims(data[:, self.body_part, 2:-1:4], axis=-1)
            #     proc_x = np.concatenate((s2, s3, s4), axis=-1)
            

            if 'x' in self.scale.lower():
                proc_x= self.preprocessing_X(proc_x)
            train_y.append(a[1])
            train_x.append(proc_x)
            print('\r Load DATA {} / {}'.format(i, len(heat_npy)), end='')
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

        return y_train

    def __len__(self):
        assert len(self.train_x) == len(self.train_y)
        return len(self.train_y) 
    
    def __getitem__(self, index):
        
        # if self.no_scale:
        #     data = self.train_x[index]
        #     label = self.train_y[index]


        data = np.transpose(self.train_x[index], (2, 0, 1))
        data = np.expand_dims(data, axis=-1)
        label = self.train_y[index]

        return data, label
    



class NRC_Data_set(Dataset):
    def __init__(self, dir, no_scale=None):

        self.y_save=[]
        self.ppc = ppc.FrameProcessWrapper('max', 'repeat')
        self.num_channel = 3
        self.dir = dir
        self.no_scale=no_scale
        self.body_part = self.body_parts()   
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()    
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 100
        self.new_label = []
        self.train_x, self.train_y= self.import_dataset()
        self.num_joints = len(self.body_part)
        # self.scaled_x, self.scaled_y = self.preprocessing()
                
    def body_parts(self):
        # body_parts = [0, 1, 2, 5, 6, 7, 8, 9, 10,
        #             91, 93, 95, 96, 99, 103, 107, 111, 
        #             112, 114, 116, 117, 120, 124, 128, 132]

        body_parts = [0, 5, 6, 7, 8, 9, 10,
                93, 95, 96, 99, 100, 103, 104, 107, 108, 111,
                114, 116, 117, 120, 121, 124, 125, 128, 129, 132]

        return body_parts
    
    def import_dataset(self):
        heat_npy = pd.read_csv(self.dir).values.tolist()
        heat_npy.sort()

        train_x = []
        train_y = []
        for i, a in enumerate(heat_npy):
        
            # data = np.load(a[0])[:,self.body_part,:].reshape(-1, 75)
            # data = self.sc1.fit_transform(data).reshape(-1, 25, 3)
            data = np.load(a[0])
            data = self.sc1.fit_transform(data.reshape(data.shape[0], -1))
            data = data.reshape(data.shape[0], 133, -1)


            if data.shape[1] * data.shape[2] == 532:
                data = data[:,:,:3:]
            
            data = self.ppc.doPreProc(data[:,self.body_part,:])#zero padding 영향 제외 위해 마지막에

            train_x.append(np.transpose(data, (2,0,1)))

            label = a[1]*100 
            train_y.append(label)
            
            print('\r Load DATA {} / {}'.format(i, len(heat_npy)), end='')
        self.y_save = train_y
        train_y = self.sc2.fit_transform(np.reshape(train_y,(-1,1)))
        return train_x, train_y
            
    # def preprocessing(self):        
    #     y_train = np.reshape(self.train_y,(-1,1))
    #     X_train = np.array(self.train_x).reshape(-1, 75)
    #     X_train = self.sc1.fit_transform(X_train)         
    #     y_train = self.sc2.fit_transform(y_train)


    #     X_train = self.train_x.reshape(-1, 25, 3)
    #     return X_train, y_train
    

    def __len__(self):
        return len(self.train_y)
    
    def __getitem__(self, index):
        
        data = self.train_x[index]
        label = self.train_y[index]
        return data, label

