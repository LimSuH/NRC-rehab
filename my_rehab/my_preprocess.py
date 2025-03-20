import os
import sys
import argparse

import pandas as pd
import csv

import cv2 as cv
import numpy as np
import pickle as pkl

from collections import Counter
from operator import itemgetter, attrgetter

class utils():
    def __init__(self):
        print("Funtions of this class is:")
        self.classes = [Normalization, FrameProcessWrapper, frameProcess, PreProc, Annotation, AngleAnnotation, Visualization, concat]
        for cls in self.classes:
            print(cls.__name__)
    def __call__(self):
        print("Funtions of this class is:")
        for cls in self.classes:
            print(cls.__name__)

class Normalization:
    def __init__(self, path='/dataset/KETI_SignLanguage/Keypoints-removal', whole=27, do=False):
        self.KSL_NP_PATH = path
        self.do = do

        if whole==53:
            self.BODY_WHOLE = 11
            self.LEFT_WHOLE = 32
            self.RIGHT_WHOLE = 53
            self.keypoint_num = 53
            
            self.SKELETONS =np.concatenate(([0,1,2,3,4,5,6,7,8,9,10],
                                [91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111],
                                [112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132]), axis=0)

        elif whole==27:
            self.BODY = 7
            self.LEFT = 17
            self.RIGHT = 27
            self.keypoint_num = 27

            self.SKELETONS =np.concatenate(([0,5,6,7,8,9,10], 
                                [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0)
        
        else:
            self.BODY = 23
            self.FACE = 91
            self.LEFT = 112
            self.RIGHT = 133
            self.keypoint_num = 133
            self.SKELETONS =np.arange(self.keypoint_num)
        
    
        
    def __call__(self, video):

        if self.do:
            print("yes")
            try:
                # video = video[:,self.SKELETONS,:]
                body = video[:, :self.BODY, :]
                left =video[:, self.BODY:self.LEFT, :]
                right =video[:, self.LEFT:self.RIGHT, :]
                
                #Do I object - 2D normalization?
                body_min_x= np.min(body[:,:,0])
                body_min_y= np.min(body[:,:,1])
                body_max_x= np.max(body[:,:,0])
                body_max_y= np.max(body[:,:,1])

                left_min_x = np.min(left[:,:,0])
                left_min_y = np.min(left[:,:,1])
                left_max_x = np.max(left[:,:,0])
                left_max_y = np.max(left[:,:,1])

                right_min_x = np.min(right[:,:,0])
                right_min_y = np.min(right[:,:,1])
                right_max_x = np.max(right[:,:,0])
                right_max_y = np.max(right[:,:,1])


                video[:, :self.BODY, 0] = (body[:,:,0] - body_min_x) / (body_max_x - body_min_x + 0.000001)
                video[:, :self.BODY, 1] = (body[:,:,1] - body_min_y) / (body_max_y - body_min_y + 0.000001)

                video[:, self.BODY:self.LEFT, 0] = (left[:,:,0] - left_min_x) / (left_max_x - left_min_x + 0.000001)
                video[:, self.BODY:self.LEFT, 1] = (left[:,:,1] - left_min_y) / (left_max_y - left_min_y + 0.000001)

                video[:, self.LEFT:self.RIGHT, 0] = (right[:,:,0] - right_min_x) / (right_max_x - right_min_x + 0.000001)
                video[:, self.LEFT:self.RIGHT, 1] = (right[:,:,1] - right_min_y) / (right_max_y - right_min_y + 0.000001)
            
                return video
        
            except:
                return video
            
        return video
        

class FrameProcessWrapper:
    #비디오마다 다른 프레임을 통일하기 위한 class 입니다.
    #기준 프레임 수를 사용자로부터 받아 길면 cutting, 길면 padding을 합니다.
    #padding의 경우 zero(0으로 된 배열 추가), last(마지막 프레임 추가), repeat(프레임 반복) 세가지 종류를 선택할 수 있습니다.
    #기본 option: avg(109 frame), repeat padding
    def __init__(self, cutting='avg', padding='repeat'):

        if cutting == 'avg':
            self.ceiling = 128
        elif cutting == 'max':
            self.ceiling = 283
        elif cutting == 'most':
            self.ceiling = 97

        if padding == 'zero':
            self.padding = frameProcess.zero_padding
        elif padding == 'last':
            self.padding = frameProcess.last_padding
        elif padding == 'repeat':
            self.padding = frameProcess.repeat_padding

    def doCutting(self, origin_np):
        return frameProcess.cutting(origin_np, self.ceiling)
    
    def doPadding(self, origin_np):
        return self.padding(origin_np, self.ceiling)
    
    def doPreProc(self, origin_np):

        if origin_np.shape[0] > self.ceiling:#cutting
            return self.doCutting(origin_np)
        elif origin_np.shape[0] < self.ceiling:#Padding
            return self.doPadding(origin_np)
        else:
            return origin_np
            
class frameProcess:
    #데이터마다 다른 프레임을 통일시키기 위한 함수입니다.
    def cutting(origin, ceiling):
        
        return origin[:ceiling]

    def zero_padding(origin, ceiling, land):
        pad_np = np.zeros((ceiling - origin.shape[0], land, 3), dtype=np.float32)
        result_np = np.concatenate((origin, pad_np), axis=0)
    
        return result_np

    def last_padding(origin, ceiling, land):
        last_np = origin[-1].reshape(1, land, 3)
        extension = ceiling - origin.shape[0]

        for i in range(extension):
            origin = np.concatenate((origin, last_np), axis=0)
        
        return origin

    def repeat_padding(origin, ceiling):
        i = 0
        
        while origin.shape[0] < ceiling:
            repeat_np = origin[:ceiling - origin.shape[0]]
            origin = np.concatenate((origin, repeat_np), axis=0)
            

        return origin


class PreProc:
    #데이터 split을 위한 class
    #전체 데이터경로, 데이터 라벨, subset 갯수를 지정하여 실행하면 training:validation = 7:3 으로, class별로 동일한 갯수로 (video_name, label) 형태로 쓰여진 정보 파일을 만듭니다.
    #최종 결과물은 csv파일로, [OUTPUTPATH]_train.csv, [OUTPUTPATH]_test.csv 두가지 파일이 저장됩니다.
    #subset을 지정하지 않고 전체 데이터에 대해서도 실행 가능합니다.
    #*aug: augmentation 해서 파일 이름 형태가 기존이랑 좀 다른 파일이 있었음, 그거 쓰려고 한거...신경x

    def __init__(self, dir='', csv='', out='', subsetNum =-1, rate=0.75, aug=False):
    
        self.INPUTPATH = dir
        self.csv = csv
        self.OUTPUTPATH = out
        self.subsetNum = subsetNum
        self.rate = rate
        self.MODE = ['train', 'test']
        
        self.annotation_file = pd.read_csv(self.csv).values.tolist()
        #label index mapping
        
        self.annotation_file= dict(sorted(dict(self.annotation_file).items(), key=lambda x: x[1]))
        self.npy_list = os.listdir(self.INPUTPATH)
        self.aug=aug

    def __call__(self):
       
        self.subset_num = int(len(self.annotation_file) * self.rate)


        for mode in self.MODE:

            label_path = self.OUTPUTPATH + mode + '.csv'
            label_file = open(label_path, 'w', encoding='utf-8', newline='')
            lable_wr = csv.writer(label_file)
            classes = len(set(self.annotation_file.values()))#클래스갯수
            classNum = int(self.subset_num / classes)
            mode_anno = {}
            c = 0
            index = 0

            for video, label in self.annotation_file.items():
            
                if c >= classNum:
                    if label == index:
                        continue
                    else:
                        c = 0
                        index += 1

                mode_anno[video] = label
                c+=1

            for video in mode_anno:
                del self.annotation_file[video]

            # make label and preproc
            print('{} : now processing on frames...'.format(mode))
        
            i = 0
            result_numpy = []
            lable_wr.writerow(['video_name', 'label_index'])
            
            mode_num = len(mode_anno)
            '''
            tgname: neuron2와 neuronS1의 파일명 차이때문에 문제 있음,
            neuron2에 있는 npy파일은 죄다 동영상 확장자+npy
            '''
            for target in self.npy_list:
                if self.aug:
                    tgname = target[:-4]
                else:
                    tgname = target[:-8]
                if i >= mode_num:
                    break
                    
                try:
                    label_index = mode_anno[tgname]
                except(KeyError):
                    continue

                lable_wr.writerow([target, label_index])
                #data_numpy = np.load(os.path.join(self.INPUTPATH, target))

                # result_numpy.append(preprocInst.doPreProc(data_numpy))
                i+=1
                print('\r{} ({} / {})'.format(tgname, i, mode_num), end='')
                del mode_anno[tgname]
                

            # result_numpy = np.array(result_numpy)
            # np.save(OUTPUTPATH + '_' + args.C + args.P + '.npy', result_numpy)

            label_file.close()

            check_annotation = dict(sorted(pd.read_csv(label_path, names=['video_name', 'label_index']).values.tolist()))
            label = list(check_annotation.values())

            print('\n')
            #print(label)
            print(Counter(label))
            print('\n\n')

        train_split = pd.read_csv(self.OUTPUTPATH + self.MODE[0]+'.csv')
        test_split = pd.read_csv(self.OUTPUTPATH + self.MODE[1]+'.csv')
        return train_split, test_split


class Annotation:
    #위의 PreProc의 결과물로 만들어진 csv 파일을 넣어, pyskl 학습에 사용되는 형태의 최종 데이터를 생성합니다.(pkl 파일)

    def __init__(self, np_path, split_path, out_path, save_ann=False, whole=False, aug=False):
        self.mode=['train', 'test']
        self.anno_path = [split_path + m + '.csv' for m in self.mode]
        self.np_path = np_path
        self.split_path = split_path #just common part of train / test label file name
        self.out_path=out_path
        self.kp_list = {}
        self.npy_list = pd.read_csv(self.anno_path[0], index_col=1).values.tolist() + pd.read_csv(self.anno_path[1], index_col=1).values.tolist()
        self.split = {}
        self.anno={}
        self.whole = whole
        self.norm = Normalization(whole=self.whole, do=False)
        self.keypoint = self.norm.SKELETONS
        self.keypoint_num = self.norm.keypoint_num
        self.frame_num = 128
        self.save = save_ann
        self.aug = aug

    def __call__(self):
        self.load_data()
        
        return self.make_anno()

    
    def load_data(self):
        for i, n in enumerate(self.npy_list):
            print("\rload npy files {}/{}...".format(i + 1, len(self.npy_list)), end='')
            a_npy = np.load(os.path.join(self.np_path, n[0]))
            if self.aug:
                    self.kp_list[n[0][:-4]] = np.expand_dims(a_npy, axis=0)
            else:
                self.kp_list[n[0][:-8]] = np.expand_dims(a_npy, axis=0)

        print('Done')
        
        for m in self.mode:
            print("load split names. Mode: {}".format(m), end=' ...')
            split_names = pd.read_csv(self.split_path + m + '.csv')
            self.split[m] = list(split_names['video_name'])
            self.split[m+'_label'] = list(split_names['label_index'])
            print('Done')
        print('\n')



    def make_anno(self):
        self.anno['split'] = {'train': self.split['train'], 'test':self.split['test']}

        anno_list = []

        for m in self.mode:
            print("Mode: {}".format(m))
            idx = 1
            for s, l in zip(self.split[m], self.split[m+'_label']):
                print("\rmaking annotation array...{}/{}".format(idx, len(self.split[m])), end='')
                if self.aug:
                    npy = self.kp_list[s]

                else:
                    npy = self.kp_list[s[:-8]]


                try:
                    npy = npy[:,:,self.keypoint,:]#extract keypoints
                    npy[0] = self.norm(npy[0])
                    keypoint = npy[..., :2]
                    score = npy[..., 2]
                except:
                    keypoint = np.zeros((1, self.frame_num, self.keypoint_num, 2), dtype=np.float32)
                    score = np.zeros((1, self.frame_num, self.keypoint_num), dtype=np.float32)

                anno_list.append({'frame_dir': s, 'label': l, 'img_shape':(1080, 1920), 'original_shape':(1080, 1920), 
                            'total_frames':keypoint.shape[1], 'num_person_raw':1, 'keypoint':keypoint, 'keypoint_score':score})
                idx +=1
            print(' ')

        self.anno['annotations'] = anno_list

        if self.save:
            with open(os.path.join(self.out_path), 'wb') as f:
                pkl.dump(self.anno, f)
            print("annotation file save to :", os.path.join(self.out_path))

        return self.anno

class AngleAnnotation:

    def __init__(self, np_path, split_path, out_path, whole):
        self.mode=['train', 'test']
        self.anno_path = [split_path + m + '.csv' for m in self.mode]
        self.np_path = np_path
        self.split_path = split_path #just common part of train / test label file name
        self.out_path=out_path
        self.kp_list = {}
        self.npy_list = pd.read_csv(self.anno_path[0], index_col=1).values.tolist() + pd.read_csv(self.anno_path[1], index_col=1).values.tolist()
        self.split = {}
        self.anno={}
        self.keypoint_num = 27
        self.frame_num = 109


    def __call__(self):
        self.load_data()
        self.make_anno()

    
    def load_data(self):
        for i, n in enumerate(self.npy_list):
            print("\rload npy files {}/{}...".format(i + 1, len(self.npy_list)), end='')
            a_npy = np.load(os.path.join(self.np_path, n[0]))
            self.kp_list[n[0][:-8]] = np.expand_dims(a_npy, axis=0)

        print('Done')
        
        for m in self.mode:
            print("load split names. Mode: {}".format(m), end=' ...')
            split_names = pd.read_csv(self.split_path + m + '.csv')
            self.split[m] = list(split_names['video_name'])
            self.split[m+'_label'] = list(split_names['label_index'])
            print('Done')
        print('\n')



    def make_anno(self):
        self.anno['split'] = {'train': self.split['train'], 'test':self.split['test']}

        anno_list = []

        for m in self.mode:
            print("Mode: {}".format(m))
            idx = 1
            for s, l in zip(self.split[m], self.split[m+'_label']):
                print("\rmaking annotation array...{}/{}".format(idx, len(self.split[m])), end='')

                try:
                    keypoint = self.kp_list[s[:-8]]
                except:
                    keypoint = np.zeros((1, self.frame_num, self.keypoint_num), dtype=np.float32)

                anno_list.append({'frame_dir': s, 'label': l, 'img_shape':(720, 1280), 'original_shape':(720, 1280), 
                            'total_frames':keypoint.shape[1], 'num_person_raw':1, 'keypoint':keypoint})
                idx +=1
            print(' ')

        self.anno['annotations'] = anno_list

        with open(os.path.join(self.out_path), 'wb') as f:
                pkl.dump(self.anno, f)
        print("annotation file save to :", os.path.join(self.out_path))


class concat:
    def __init__(self, an_path1, an_path2, out_path, whole=False):
        self.an_path1 = an_path1
        self.an_path2 = an_path2
        self.out_path = out_path
        self.anno1 = ''
        self.anno2 = ''

    def __call__(self):
        self.load_anno()
        self.concat_anno()
       
    def load_anno(self):
        with open(self.an_path1, 'rb') as f:
            self.anno1 = pkl.load(f)

        with open(self.an_path2, 'rb') as f:
            self.anno2 = pkl.load(f)

    def concat_anno(self):
        train1 =self.anno1['split']['train']
        train2 =self.anno2['split']['train']
        self.anno1['split']['train'] = train1 + train2

        test1 = self.anno1['split']['test']
        test2 = self.anno2['split']['test']
        self.anno1['split']['test'] = test1+test2

        ans1 = self.anno1['annotations']
        ans2 = self.anno2['annotations']
        self.anno1['annotations'] = ans1+ans2

        with open(self.out_path, 'wb') as f:
            pkl.dump(self.anno1, f)


class Visualization:
    #generative 모델로 만들어진 npy 데이터를 시각화하여 비디오로 만드는 파일
    #video가 생성되나, 코덱의 오류로 재생되지 않는 이슈가 있습니다-수정 필요

    def __init__(self, video = None, keypoint = None, out_path = './data/video/generated_video.avi', frame_path = './data/frame/', size =(1080, 1920), fps = 30):
        self.video_path = video
        self.keypoint = keypoint.numpy()
        self.out_path = out_path
        self.frame_path = frame_path
        if os.path.exists(self.out_path) == False:
            os.mkdir(self.out_path)
        if os.path.exists(frame_path) == False:
            os.mkdir(self.frame_path)

        self.frame = keypoint.shape[0]
        self.size= size
        self.fps = fps
        self.fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.canvas = np.zeros((self.frame, 1080, 1920, 3), dtype = np.uint8)

    def __call__(self):
        videoWriter = cv.VideoWriter(self.out_path, cv.VideoWriter_fourcc('M','J','P','G'), 30, (1080, 1920))
        print(self.canvas.shape)
        for i, f in enumerate(self.keypoint):
            print("\rnow visualize the generated video...{}/{}".format(i+1, self.frame), end=' ')
            img = self.canvas[i]
            for k in f:
                img = cv.circle(img, (int(k[0]), int(k[1])), 3, (255, 255, 255), -1)
            cv.imwrite(self.frame_path + str(i) + '.png', img)
            img=cv.imread(self.frame_path + str(i) + '.png')
            videoWriter.write(img)
        print('Done.')
        videoWriter.release()


        v