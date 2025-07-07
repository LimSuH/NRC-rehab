'''
you don't need to look this code. just ignore
'''

import torch
from torch import nn
from torch.nn import LSTM



class Sgcn_Lstm(nn.Module):
    def __init__(self, input_dim, bias_mat_1, bias_mat_2):
        super(Sgcn_Lstm, self).__init__()
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.num_joints = 25

        self.TC_block1 = nn.Sequential(
            nn.Conv2d(input_dim, 64, (9,1), padding="same") # _,_,25,3
        )

        self.hop_block1 = nn.Sequential(
            nn.Conv2d(input_dim+64, 64, (1,1)),
            nn.ReLU()
        )

        self.hop_block2 = nn.Sequential(
            nn.Conv2d(input_dim+64, 64, (1,1)),
            nn.ReLU()
        )


        self.convLSTM1 = nn.Sequential(
            ConvLSTM(64, 25, kernel_size=(1,1), num_layers=1,
                 batch_first=False, bias=True, return_all_layers=False),
            nn.Tanh()

        )
        
        self.temporal_conv1 = nn.Sequential(
            nn.Conv2d(128, 16, (9,1), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.temporal_conv2 = nn.Sequential(
            nn.Conv2d(16, 16, (15,1), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.temporal_conv3 = nn.Sequential(
            nn.Conv2d(16, 16, (20,1), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        # tensorflow: (batch, height, width, channels) (NHWC) -> torch: (NCHW) 
        tc1 = self.TC_block1(x)# _,3,_25 -> _,64,_,25
        first_temporal = torch.cat([x, tc1], dim=1)# _,64,_,25 ->  _,67,_25

        """first hop localization"""
        h1_0 = self.hop_block1(first_temporal)# _,67,_,25 ->  _,64,_25
        h1 = h1_0.unsqueeze(1)# _,67,_,25 ->  _,1,64,_25
        hc1 = self.convLSTM1(h1)#  _,1,25,_25
        f_1 = hc1[:,0,:,:,:]#  _,25,_25
        logits = f_1.permute(0,2,1,3)
        
        sfm1 = nn.Softmax(dim=-1)
        coefs = sfm1(self.leakyReLU(logits)+self.bias_mat_1) # pytorch shape, +[25,25]
        coefs = coefs.permute(0,2,1,3)
        gcn_x1 = torch.einsum('nwtv,nctw->nctv', coefs, h1_0)# _,64,_,25

        """second hop localization"""
        h2_0 = self.hop_block2(first_temporal)# _,_,25,64
        h2 = h2_0.unsqueeze(1)# _,_,25,1,64
        hc2 = self.convLSTM1(h2)# _,_,25,1,25
        f_2 = hc2[:,0,:,:,:]
        logits = f_2.permute(0,2,1,3)

        sfm2 = nn.Softmax(dim=-1)
        coefs = sfm2(self.leakyReLU(logits)+self.bias_mat_2)
        coefs = coefs.permute(0,2,1,3)
        gcn_y1 = torch.einsum('nwtv,nctw->nctv', coefs, h2_0)
        # _,_,25,64

        gcn_1 = torch.cat([gcn_x1, gcn_y1], dim=1)# _,128,_,25

        """Temporal convolution"""
        z1 = self.temporal_conv1(gcn_1)
        z2 = self.temporal_conv2(z1)
        z3 = self.temporal_conv3(z2)
        z = torch.cat([z1, z2, z3], dim=1)

        return z

        


#https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))



class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        #return layer_output_list, last_state_list
        return layer_output_list[0]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class GetLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out
       
class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstmS = nn.Sequential(
            nn.LSTM(1200, 80, dropout=0.25),
            GetLSTMOutput(),
            nn.LSTM(80, 40, dropout=0.25),
            GetLSTMOutput(),
            nn.LSTM(40, 40, dropout=0.25),
            GetLSTMOutput(),
            nn.LSTM(40, 80, dropout=0.25)
        )
        self.lin = nn.Sequential(
            nn.Linear(283*80, 1)
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, x.shape[0], x.shape[1]*x.shape[-1])) # _,48,_,25 time,b,1200
        # out = self.lstm(x.shape[-1], 80, x, return_sequences=True)
        # out = self.lstm(out.shape[-1], 40, out, return_sequences=True)
        # out = self.lstm(out.shape[-1], 40, out, return_sequences=True)
        # h_n, c_n = self.lstm(out.shape[-1], 80, out)

        h_n, c_n = self.lstmS(x)
        last=h_n.transpose(0,1)
        last = last.reshape(-1, last.shape[1]*last.shape[-1])

        lin = self.lin(last)
        return lin
    
    
class train_network(nn.Module):
    def __init__(self,
                 input_dim = [3, 48, 48],
                 bias_mat_1=None,
                 bias_mat_2=None):
        super().__init__()
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2

        self.sgcn1 = Sgcn_Lstm(input_dim[0], bias_mat_1, bias_mat_2)
        self.sgcn2 = Sgcn_Lstm(input_dim[1], bias_mat_1, bias_mat_2)
        self.sgcn3 = Sgcn_Lstm(input_dim[2], bias_mat_1, bias_mat_2)
        self.lstm = Lstm()
        

    def forward(self, input):
        x = self.sgcn1(input)
        y = self.sgcn2(x)
        y = y + x
        z = self.sgcn3(y)
        z = z + y
        out = self.lstm(z)

        return out
    
# class traincell(nn.Sequential):
#      def __init__(self,
#                  AD=None,
#                  AD2=None,
#                  bias_mat_1=None,
#                  bias_mat_2=None):
        
#         self.AD = AD
#         self.AD2 = AD2
#         self.bias_mat_1 = bias_mat_1
#         self.bias_mat_2 = bias_mat_2


#         super().__init__(
#                 train_network(input_dim=[3,48,48],bias_mat_1=self.bias_mat_1, bias_mat_2=self.bias_mat_2)
#             )