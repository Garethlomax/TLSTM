# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script 

"""

import torch
import torch.nn
import torch.nn.functional as F

class TLSTM(nn.Module):


class TLSTM_UNIT(nn.Module):
    """Base unit for T_lstm
    """
    
    def __init__(self, input_size, hidden_size):
        
        # 
        super(TLSTM_UNIT, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        
        # initialise weights for NNs.
        # TODO: why not use linnear layers? 
        # TODO: simplify into fewer layers - can superimpose
        
        self.Wi = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.Ui = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        # TODO: add biases 
        
        self.Wf = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.Uf = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        # TODO: add biases 
        
        # output
        self.Wog = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.Uog = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        # TODO: add biases 
        
        self.Wc = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.Uc = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        # TODO: add biases 
        
        # TODO: figure out what issue is with 
        self.W_decomp = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        # TODO: add biases
        
        self.WoFC = nn.Parameter(torch.Tensor(self.hidden_size, self.fc_size))
        # TODO: add biases
        
        
        
        
        
        
    def forward(self,x, hidden):
        

