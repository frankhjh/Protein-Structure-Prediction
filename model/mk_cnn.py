#!/usr/bin/env python
import torch
import torch.nn as nn

class multi_kernel_cnn(nn.Module):
    def __init__(self,vocab_size,max_len,padding_idx,num_classes):
        super(multi_kernel_cnn,self).__init__()
        self.vocab_size=vocab_size
        self.text_len=max_len
        self.padding_idx=padding_idx
        self.num_classes=num_classes
        self.embedd_dim=128
        self.feature_size=128 
        self.window_sizes=[2,3,4,6,8,9,10,20,50,100]
        self.embedding_layer=nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embedd_dim,
                                          padding_idx=padding_idx)
        self.convs=nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=self.embedd_dim,
                                           out_channels=self.feature_size,
                                           kernel_size=self.window_sizes[i],
                                           stride=1),
                                  nn.BatchNorm1d(self.feature_size),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=self.text_len-self.window_sizes[i]+1)) 
                                  for i in range(len(self.window_sizes))])
        self.dropout=nn.Dropout(p=0.2)
        self.fc1=nn.Linear(in_features=len(self.window_sizes)*self.feature_size,
                          out_features=512)
        self.fc2=nn.Linear(in_features=512,out_features=self.num_classes)
    def forward(self,x): 
        embedded_x=self.embedding_layer(x) # shape:[batch_size,text_len,embedd_dim]
       
        embedded_x=embedded_x.permute(0,2,1) # shape:[batch_size,embedd_dim,text_len]
       
        conv_x=[conv(embedded_x) for conv in self.convs] # size(conv_x[i])=[batch_size,feature_size,1]
        
        x=torch.cat(conv_x,dim=1) # shape:[batch_size,10*feature_size,1]
        x=x.view(-1,x.size(1)) # drop final dim 1
        
        x=self.dropout(x)
        x=self.fc1(x)
        x=self.dropout(x)
        out=self.fc2(x)
        return out    