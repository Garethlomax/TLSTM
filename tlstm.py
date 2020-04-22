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
    
    def __init__(self, input_size, hidden_size, bias = True, time_delay_type = 'log'):
        
        # 
        super(TLSTM_UNIT, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.bias = bias
        self.time_delay_type = time_delay_type
        self.tanh = nn.tanh
        # initialise weights for NNs.
        # TODO: why not use linnear layers? 
        # TODO: simplify into fewer layers - can superimpose
        # TODO: reframe to two linnear operations
        
        
        # TODO: get rid of half of the biases. 
        self.Wi = nn.linnear(self.input_size, self.hidden_size, bias = bias)
        self.Ui = nn.linnear(self.hidden_size, self.hidden_size, bias = bias)
        
        self.Wf = nn.linnear(self.input_size, self.hidden_size, bias = bias)
        self.Uf = nn.linnear(self.hidden_size, self.hidden_size, bias = bias)

        #output
        self.Wog = nn.linnear(self.input_size, self.hidden_size, bias = bias)
        self.Uog = nn.linnear(self.hidden_size, self.hidden_size, bias = bias)
        
        
        self.Wc = nn.linnear(self.input_size, self.hidden_size, bias = bias)
        self.Uc = nn.linnear(self.hidden_size, self.hidden_size, bias = bias)
        
        # TODO: figure out what issue is with 
        self.W_decomp = nn.linnear(self.hidden_size, self.hidden_size, bias = bias)
        
        self.WoFC = nn.linnear(self.hidden_size, self.fc_size, bias = bias)
        
        # TODO: figure out what softmax layer output is
        # self.W_softmax = self.init_weights(fc_dim, output_dim, name='Output_Layer_weight',reg=None)
                                               
        
        
        
    def time_delay(self, t):
        """ calculates time delay between input samples according to different
        rules"""
        
        if time_delay_type == 'log': 
            #TODO: redo this with pytorch constants ect to make static and faster
            
            T = 1 / np.log(np.e + t)
            
            T_vec = torch.full(self.hidden_size, fill_value = T)
            # TODO: check if needs requires grad for differentiability.
            
            return T_vec
        
        elif time_delay_type == 'reciprocal':
            
            T_vec = torch.full(self.hidden_size, fill_value = 1/ t)
            
            return T_vec
        
    
        
        
        
        
    def forward(self,x, t, hidden, cell):
        """hidden and cell are previous hidden memory and cell state"""
        
        T = self.time_delay(t)
        
        # if there is a time delay decompose by time delay
        C_ST = self.tanh(self.W_decomp(cell))
        # discount short term cell memory
        C_ST_dis = T * C_ST
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

