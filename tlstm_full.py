#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:31:32 2020

@author: garethlomax
"""


import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

from tlstm import TLSTM_UNIT

class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, bias = True, time_delay_type = 'log'):
        super(TLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.bias = bias
        self.time_delay_type = time_delay_type
        self.num_layers = len(hidden_sizes)
        cell_list = []
        
        for i in range(self.num_layers):
            input_dim = self.input_size if i == 0 else self.hidden_sizes[i-1]
            
            cell_list.append(TLSTM_UNIT(input_size = input_dim, hidden_size = hidden_sizes[i], bias = bias, time_delay_type = time_delay_type))
            
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, x, hidden_state_copy = None):
        # TODO: hidden state crap 
        
        layer_output_list = []
        last_state_list = []
        
        seq_len = x.size(1)
        cur_layer_input = x
        # add in hidden_state record.
        # use reset params function.
        
        # we reset h at every timestep. h is initialised as zeros and not gaussian
        h_shape = list(x.shape[:1] + x.shape[2:]) # seq is second
        hidden_state = [[torch.zeros(h_shape), torch.zeros(h_shape)] for _ in range(self.num_layers)]
        if hidden_state_copy == None:
            hidden_state = [[torch.zeros(h_shape), torch.zeros(h_shape)] for _ in range(self.num_layers)]
        else:
            hidden_state = hidden_state_copy
        #Main iteration loo[]
        for i in range(self.num_layers):
            h, c = hidden_state[i] # add hidden state
            # reset hidden state 
            output_inner = []
            for j in range(seq_len):
                # put t in here - how do we get out of dataset??. 
                # include in the array? 
                
                h, c = self.cell_list[i](cur_layer_input[:,j], h, c)

                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim = 1) # check layer output and dimensions
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h,c))
           
        layer_output = layer_output_list[-1]
        
        # Layer output is stacked tensor of h from final 
        # Last state list is last hidden state and cell memory for each layer
        return layer_output, last_state_list
    
class Decoder(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, bias = True, time_delay = 'log'):
        super(Decoder, self).__init__()
        
        self.TLSTM = TLSTM(input_size, hidden_size, bias, time_delay)
        self.output = nn.Linear(hidden_size, output_size, bias = bias)
        
        
        
   
    def forward(self, x, hidden_state):
        x, hidden_state = self.TLSTM(x, hidden_state)
        x = self.output(x)
        return x, hidden_state
    
class Encoder(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, bias = True, time_delay = 'log'):
        super(Encoder, self).__init__()
        
        self.TLSTM = TLSTM(input_size, hidden_size, bias, time_delay)
        
        def forward(x):
            
            # we do not need the h outputs - these are not actually important
            # in the encoder
            _, last_state_list = self.TLSTM(x)
            
            return last_state_list

class AE(nn.Module):
    def __init__(self,enc_input_size, enc_hidden_sizes, dec_input_size, dec_hidden_sizes, bias = True, time_delay_type = 'log', copy_over = True):
        super(AE, self).__init__()
        
        self.enc_input_size = enc_input_size
        self.enc_hidden_sizes = enc_hidden_sizes
        self.dec_input_size = dec_input_size
        self.dec_hidden_sizes = dec_hidden_sizes
        self.bias = bias
        self.time_delay_type = time_delay_type
        self.copy_over = copy_over
        
        
        
        self.encoder = TLSTM(self.enc_input_size, self.enc_hidden_sizes, self.bias, self.time_delay_type)
        self.decoder = TLSTM(self.dec_input_size, self.dec_hidden_sizes, self.bias, self.time_delay_type)
        
    def forward(self, x):
        """ x is batch * seq * features"""
        #Initialising output array to place decoder output into
        outputs = torch.zeros_like(x)
        

        #Extracting hidden state from encoder
        seq_len = x.shape[1]
        hidden_state = self.encoder(x)
        
        #Initialising input array to send into Decoder
        #We decode one step at a time
        init_input_shape = x.shape
        init_input_shape[1] = 1
        x = torch.zeros(init_input_shape)
        
        for i in range(seq_len):
            x, hidden_state = self.decoder(x, hidden_state)
            #TODO: we may need to take x out of list ect.
            outputs[:,i,:] = x 
            
            # we now reverse along the sequence lenght axis. 
            # as justified by seq 2 seq models. 
            

        return outputs
                
            
            
        
        
            
            
        

    
# class TLSTMAE(nn.Module):
#     def __init__(self,enc_input_size, enc_hidden_sizes, dec_input_size, dec_hidden_sizes, bias = True, time_delay_type = 'log', copy_over = True):
#         super(TLSTMAE, self).__init__()
#         """ By default only last state is copied over"""
#         self.enc_input_size = enc_input_size
#         self.enc_hidden_sizes = enc_hidden_sizes
#         self.dec_input_size = dec_input_size
#         self.dec_hidden_sizes = dec_hidden_sizes
#         self.bias = bias
#         self.time_delay_type = time_delay_type
#         self.copy_over = copy_over
        
#         self.encoder = TLSTM(self.enc_input_size, self.enc_hidden_sizes, self.bias, self.time_delay_type)
#         self.decoder = TLSTM(self.dec_input_size, self.dec_hidden_sizes, self.bias, self.time_delay_type)
        
#     def forward(self, x):
#         output_list = []
#         x, last_state = self.encoder(x)
#         output = encoded
#         # x is of dimensions batch, seq, features 
#         #TODO: sort output of tlstm and fix this
#         zeros = torch.zeros_like(output)
        
#         output_list.append(encoded)
#         for i in range(x.shape[1]): #i.e sequence length
#             x = self.decoder(encoded)
#             output_list.append(x)
#             encoded = x
#         return x, encoded
        
        
        
        
            