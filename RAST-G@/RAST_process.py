import numpy as np

class FrameProcessWrapper:
    #비디오마다 다른 프레임을 통일하기 위한 class 입니다.
    #기준 프레임 수를 사용자로부터 받아 길면 cutting, 길면 padding을 합니다.
    #padding의 경우 zero(0으로 된 배열 추가), last(마지막 프레임 추가), repeat(프레임 반복) 세가지 종류를 선택할 수 있습니다.
    #기본 option: avg(109 frame), repeat padding
    def __init__(self, cutting='avg', padding='repeat', ceiling=None):

        if cutting == 'avg':
            self.ceiling = 128
        elif cutting == 'max':
            self.ceiling = 283
        elif cutting == 'most':
            self.ceiling = 97
        elif cutting == 'custom':
            self.ceiling = ceiling

        if padding == 'zero':
            self.padding = frameProcess.zero_padding
        elif padding == 'last':
            self.padding = frameProcess.last_padding
        elif padding == 'repeat':
            self.padding = frameProcess.repeat_padding

    def doCutting(self, origin_np):
        #return frameProcess.cutting(origin_np, self.ceiling)
        return frameProcess.frameDrop_cutting(origin_np, self.ceiling)
    
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
    def cutting(origin, ceiling):
        
        return origin[:ceiling]
    
    def frameDrop_cutting(origin, ceiling):
        orf = origin.shape[0]
        return np.array([origin[int(orf / ceiling * spf)] for spf in range(ceiling)])

    def zero_padding(origin, ceiling):
        pad_np = np.zeros((ceiling - origin.shape[0], origin.shape[1], 3), dtype=np.float32)
        result_np = np.concatenate((origin[:,:,:3], pad_np), axis=0)
    
        return result_np

    def last_padding(origin, ceiling):
        last_np = origin[-1].reshape(1, origin.shape[1], 3)
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