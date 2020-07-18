#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 21:35:02 2020

@author: garethlomax
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

import pandas as pd 
import time
import datetime
import numpy as np
from data_inf import VarLenDataloader, VarLenDataset
import pandas as pd 
import time
import datetime
import numpy as np
from data_inf import VarLenDataloader, VarLenDataset

from tlstm import TLSTM_UNIT

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from livelossplot import PlotLosses
from pycm import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True

    return True

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


"""defining cuda"""
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
# device = 'cuda'


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
        self.output_size = dec_input_size
        
        
        self.encoder = TLSTM(self.enc_input_size, self.enc_hidden_sizes, self.bias, self.time_delay_type)
        self.decoder = TLSTM(self.dec_input_size, self.dec_hidden_sizes, self.output_size, self.bias, self.time_delay_type)
        
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
        outputs = torch.flip(outputs, (1,))
        return outputs
        
        


def train(model, optimizer, criterion, data_loader):
    model.train()
    train_loss, train_accuracy = 0, 0
    for X in data_loader:
        X = X.to(device)
        optimizer.zero_grad()
        a2, z = model(X)
        #a2 = model(X.view(-1, 28*28)) #What does this have to look like for our conv-net? Make the changes!
        loss = criterion(a2, torch.max(X.long(), 1)[1])
        loss.backward()
        train_loss += loss*X.size(0)
        optimizer.step() 
        y_pred = F.log_softmax(a2, dim=1).max(1)[1]
        x_true = F.log_softmax(X, dim=1).max(1)[1]
        train_accuracy += accuracy_score(x_true.cpu().numpy(), y_pred.detach().cpu().numpy())*X.size(0)
        
    return train_loss/len(data_loader.dataset), train_accuracy/len(data_loader.dataset)
  
def validate(model, criterion, data_loader):
    model.eval()
    validation_loss, validation_accuracy = 0., 0.
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, 1, 28, 28))
            #a2 = model(X.view(-1, 28*28)) #What does this have to look like for our conv-net? Make the changes!
            loss = criterion(a2, y)
            validation_loss += loss*X.size(0)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            validation_accuracy += accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())*X.size(0)
            
    return validation_loss/len(data_loader.dataset), validation_accuracy/len(data_loader.dataset)
  
def evaluate(model, data_loader):
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, 1, 28, 28))
            #a2 = model(X.view(-1, 28*28)) #What does this have to look like for our conv-net? Make the changes!
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
            
    return np.concatenate(y_preds, 0),  np.concatenate(ys, 0)


def train_model(model, criterion, dataloader):
    set_seed(seed)
    model = model.double().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
  
    # train_loader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=0)
#   validation_loader = DataLoader(mnist_validate, batch_size=test_batch_size, shuffle=False, num_workers=0)
#   test_loader = DataLoader(mnist_test, batch_size=test_batch_size, shuffle=False, num_workers=0)
  
    liveloss = PlotLosses()
    for epoch in range(100):
        logs = {}
        train_loss, train_accuracy = train(model, optimizer, criterion, dataloader)

        logs['' + 'log loss'] = train_loss.item()
        logs['' + 'accuracy'] = train_accuracy.item()

    #   validation_loss, validation_accuracy = validate(model, criterion, validation_loader)
    #   logs['val_' + 'log loss'] = validation_loss.item()
    #   logs['val_' + 'accuracy'] = validation_accuracy.item()

        liveloss.update(logs)
        liveloss.draw()
        
        
        # if train_accuracy.item() >= 0.9985:
        #     torch.save(model.state_dict(), F"TLSTM_auto"+str(epoch)+".pth")
        #     print("peak accuracy")
        #     break
    return model

def offset_log(x):
    return np.log(x + 1)

def time_delay_days(df):
    a = df['date_start']
    b = df['date_end']
    
    a = time.mktime(time.strptime(a, "%Y-%m-%d"))
    b = time.mktime(time.strptime(b, "%Y-%m-%d"))
    c = datetime.timedelta(seconds=b-a)
    return c.days

def origin_time(strtime):
    return time.mktime(time.strptime(strtime, "%Y-%m-%d"))

def time_delay_zero(df):
    # origin = time.mktime((0.0,))
    return datetime.timedelta(seconds=df).days

def array_dict_map(dictionary, keys):
    """ for mappin embeddings from neural nets"""
    out = np.zeros((len(keys), 50))
    for i,key in enumerate(keys):
        out[i] = dictionary[key]
    return out

def embed_dict(keys, vals):
    dictionary = {key: val for key, val in zip(keys, vals)}
    return dictionary
    
def embedded_side_dict():
    sides_names = np.load("sides_names.npy")
    sides_generated = np.load("sides_generated.npy")
    return embed_dict(sides_names, sides_generated)

def embed_df_col(df, column, dictionary):
    """takes one column""" 
    new_df = array_dict_map(dictionary, df[column])
    return pd.DataFrame(new_df)

def embed_sides(df):
    dictionary = embedded_side_dict()
    df.reset_index(drop=True, inplace=True)
    df_a = embed_df_col(df, 'side_a', dictionary)
    df_b = embed_df_col(df, 'side_b', dictionary)
    df = df.drop(['side_a', 'side_b'], axis = 1)
    combined_dfs = pd.concat([df_a, df_b, df], axis = 1)
    return combined_dfs



def prod_dataset(sorting_variable = 'priogrid_gid', col_list = ['viol_1','viol_2','viol_3','priogrid_gid', 'event_count', 'id','side_a', 'side_b','duration_days','time_diff_days']):
    """ ENSURE THAT TIME DELAY IS THE LAST VARIABLE ON THE RIGHT"""

    df = pd.read_csv("../Datasets/ged191.csv")
    
    #inplace = true modifies by referenc (not exactxly but behaves like it)
    
    #datetine strptime
 
    a = time.mktime(time.strptime(df.date_start[0], "%Y-%m-%d"))
    b = time.mktime(time.strptime(df.date_end[0], "%Y-%m-%d"))
    
    c = datetime.timedelta(seconds=b-a)
    
    # def time_delay_days()
    # summarise by group 
    
    # find number of people
    
    # filter out gids with only one group
    # do we filter again? 
    df = df.groupby('priogrid_gid').filter(lambda x: len(x) > 4) # we want temporal behaviour
    df = df.groupby('priogrid_gid').filter(lambda x: len(x) < 100) # we want temporal behaviour
    
    
    # 152616
    
    
    df['log_deaths_a'] = offset_log(df.deaths_a)
    df['log_deaths_b'] = offset_log(df.deaths_b)
    df['log_best'] = offset_log(df.best)

    df['duration_days'] = df[['date_start', 'date_end']].apply(time_delay_days, axis = 1)

    # df.assign(c=df['a']-df['b']
    
    # make origional date 

    df['rel_time'] = df['date_start'].apply(origin_time)
    
    # summ = df.groupby('conflict_new_id').agg({'rel_time': 'min'})
    
    df['first_event_time'] = df.groupby(sorting_variable).rel_time.transform('min')
    # df['event_count'] = df.groupby(sorting_variable).transform('count')
    # df['event_count'] = df.groupby(sorting_variable).size()
    df['event_count'] = df.groupby(sorting_variable).priogrid_gid.transform('count')
    
    un, counts = np.unique(df.event_count, return_counts = True)
    
    number_of_seq_length = counts/un
    
    df['violence_category'] = pd.Categorical(df.type_of_violence)
    df = pd.concat([df, pd.get_dummies(df['violence_category'], prefix = 'viol')], axis=1)
    # TODO: sort by reltime first
    # This still gives us nans
    # TODO: sort out what happens in first sequence
    # not giving proper time difference - lagged by one 
    df['time_diff'] = df.sort_values(["rel_time", 'id']).groupby(sorting_variable).rel_time.diff() #TODO: sort this - id may not be the right value
    df['time_diff'] = df.time_diff.fillna(0)
    
    df['time_diff_days'] = df.time_diff.apply(time_delay_zero)
    
    df = df[col_list]
    
    dictionary = embedded_side_dict()
    
    final_df = embed_sides(df)
    return final_df







"""DATASETS"""

df = prod_dataset()

dim = len(df.keys())

#TODO: sort out inner dimensions
dataset = VarLenDataset(df)

dataloader = VarLenDataloader(dataset, max_batch_size = 50, shuffle = True)

loss = nn.MSELoss()

#TODO: choose hidden size dimension
autoencoder = AE(dim, dim//2,dim, dim//2) 

train_model(autoencoder, loss, dataloader)



 
























