#!/usr/bin/env python
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.dataset import trainset,testset
from utils.data_processor import train_processor,test_processor

# all unique tokens in train sequences
def build_token_dict(train_df):
    token_set=set()
    for i in range(train_df.shape[0]):
        seq=train_df.sequence[i]
        token_set.update([seq[j] for j in range(len(seq))])
    
    # build dict
    token_dict={'<PAD>':0,'<UNK>':1}
    for token in token_set:
        token_dict[token]=len(token_dict)
    
    return token_dict

def build_label_dict(train_df):
    # build dict
    label_dict={}
    for label in train_df.label.unique():
        label_dict[label]=len(label_dict)
    return label_dict

def max_len(train_df): #in form of DataFrame
    train_len_li=sorted(train_df.sequence.apply(lambda x:len(x)).tolist())
    return train_len_li[int(len(train_len_li)*0.98)]

def token2id(tokens,token_dict): # list of tokens
    return [token_dict.get(token,1) for token in tokens]

def label2id(label,label_dict):
    return label_dict.get(label)

def get_weight(train_data):
    label_count=train_data.groupby('label').count().to_dict()['sequence']

    idx_count=dict()
    for label,count in label_count.items():
        idx_count[label_dict[label]]=count
    
    idx_weight=dict()
    max_count=max(idx_count.values())
    for idx,count in idx_count.items():
        idx_weight[idx]=max_count/count
    
    sorted_idx_weight=sorted(idx_weight.items(),key=lambda x:x[0])
    weight_li=[i[1] for i in sorted_idx_weight]
    return torch.tensor(weight_li,dtype=torch.float)
    
def prep_dataloader(type,data,token_dict,label_dict,max_len):
    if type=='train':
        # type of data:DataFrame
        train_data_li,val_data_li=[],[]
        for ul in tqdm(data.label.unique()):
            sub_df=data[data.label==ul].reset_index(drop=True)
            val_data_li.append(sub_df.iloc[:2,:])
            train_data_li.append(sub_df.iloc[2:,:])
        
        train_data=pd.concat(train_data_li).reset_index(drop=True)
        val_data=pd.concat(val_data_li).reset_index(drop=True)

        class_weights=get_weight(train_data)

        # train data
        train_x,train_y=[],[]
        for i in tqdm(range(train_data.shape[0])):
            seq=train_data.sequence[i]
            token_li=[seq[j] for j in range(len(seq))]
            x=token2id(token_li,token_dict)
            x=x+[0]*(max_len-len(x)) if len(x)<=max_len else x[:max_len]
            y=label2id(train_data.label[i],label_dict)
            train_x.append(x)
            train_y.append(y)
        
        train_x=torch.tensor(train_x,dtype=torch.long)
        train_y=torch.tensor(train_y,dtype=torch.long)
        train=trainset(train_x,train_y)
        train_dataloader=DataLoader(train,batch_size=32,shuffle=True)
        
        # val data
        val_x,val_y=[],[]
        for i in tqdm(range(val_data.shape[0])):
            seq=val_data.sequence[i]
            token_li=[seq[j] for j in range(len(seq))]
            x=token2id(token_li,token_dict)
            x=x+[0]*(max_len-len(x)) if len(x)<=max_len else x[:max_len]
            y=label2id(val_data.label[i],label_dict)
            val_x.append(x)
            val_y.append(y)
        
        val_x=torch.tensor(val_x,dtype=torch.long)
        val_y=torch.tensor(val_y,dtype=torch.long)
        valid=trainset(val_x,val_y) # same dataset build method!
        val_dataloader=DataLoader(valid,batch_size=32,shuffle=False)

        return train_dataloader,val_dataloader,class_weights
    else:
        test_x=[]
        for i in tqdm(range(data.shape[0])):
            seq=data.sequence[i]
            token_li=[seq[j] for j in range(len(seq))]
            x=token2id(token_li,token_dict)
            x=x+[0]*(max_len-len(x)) if len(x)<=max_len else x[:max_len]
            test_x.append(x)
        test_x=torch.tensor(test_x,dtype=torch.long)
        test=testset(test_x)
        return DataLoader(test,shuffle=False)
        






        
        
        

        
        


        





