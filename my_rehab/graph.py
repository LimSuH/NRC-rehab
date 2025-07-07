'''
you don't need to look this code. just ignore
'''

import torch
import numpy as np
from IPython.core.debugger import set_trace

class Graph():
    def __init__(self, num_node):
        self.num_node = num_node
        self.AD, self.AD2, self.bias_mat_1, self.bias_mat_2 = self.normalize_adjacency()
        
    def normalize_adjacency(self):
        self_link = [(i, i) for i in range(self.num_node)]
        # neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        #                               (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        #                               (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        #                               (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        #                               (22, 23), (23, 8), (24, 25), (25, 12)]
        #NRC
        neighbor_1base=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), 
                        (5, 9), (9, 10), (10, 11), (9, 12), (12, 13), (9, 14), (9, 15), (9, 16), 
                        (6, 17), (17, 18), (18, 19), (17, 20), (20, 21), (17, 22), (17, 23), (17, 24)]
        

        neighbor_link = [(i, j) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link    
        A = np.zeros((self.num_node, self.num_node)) # adjacency matrix
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        
        A2 = np.zeros((self.num_node, self.num_node)) # second order adjacency matrix
        for root in range(A.shape[1]):
            for neighbour in range(A.shape[0]):
                if A[root, neighbour] == 1:
                    for neighbour_of_neigbour in range(A.shape[0]):
                        if A[neighbour, neighbour_of_neigbour] == 1:
                            A2[root,neighbour_of_neigbour] = 1                 
        AD = self.normalize(A)
        AD2 = self.normalize(A2)
        bias_mat_1 = np.zeros(A.shape)
        bias_mat_2 = np.zeros(A2.shape)
        bias_mat_1 = np.where(A!=0, bias_mat_1, -1e9)
        bias_mat_2 = np.where(A2!=0, A2, -1e9)
        AD = A.astype('float32')
        AD2 = A2.astype('float32')
        bias_mat_1 = bias_mat_1.astype('float32')
        bias_mat_2 = bias_mat_2.astype('float32')
        AD = torch.Tensor(AD)
        AD2= torch.Tensor(AD2)
        bias_mat_1 = torch.Tensor(bias_mat_1)
        bias_mat_2 = torch.Tensor(bias_mat_2)
        return AD, AD2, bias_mat_1, bias_mat_2
        
    def normalize(self, adjacency):
        rowsum = np.array(adjacency.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = np.diag(r_inv)
        normalize_adj = r_mat_inv.dot(adjacency)
        normalize_adj = normalize_adj.astype('float32')
        normalize_adj = torch.Tensor(normalize_adj)   
        return normalize_adj