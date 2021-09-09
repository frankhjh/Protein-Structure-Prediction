#!/usr/bin/env python
from tqdm import tqdm
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

def max_len(train_df,test_df): #in form of DataFrame
    train_max=train_df.sequence.apply(lambda x:len(x)).max()
    test_max=test_df.sequence.apply(lambda x:len(x)).max()
    return max(train_max,test_max)

def token2id(tokens,token_dict): # list of tokens
    return [token_dict.get(token,1) for token in tokens]

def label2id(label,label_dict):
    return label_dict.get(label)


def prep_dataloader(type,data,token_dict,label_dict,max_len):
    if type=='train':
        # type of data:DataFrame
        data=data.sample(frac=1.0) # shuffle
        data.reset_index(drop=True,inplace=True)
        train_data=data.iloc[:9000,:]
        val_data=data.iloc[9000:,:].reset_index(drop=True)
        
        # train_class=len(train_data.label.unique())
        # count=1
        # # make sure each category exists in the train data
        # while train_class!=245: # 245:total number of labels
        #     data.sample(frac=1.0).reset_index(drop=True,inplace=True)
        #     train_data=data.iloc[:9000,:]
        #     val_data=data.iloc[9000:,:].reset_index(drop=True)

        #     train_class=len(train_data.label.unique())
        #     count+=1
        # print(f'>>Split Done in {count} trial!')

        # train data
        train_x,train_y=[],[]
        for i in tqdm(range(train_data.shape[0])):
            seq=train_data.sequence[i]
            token_li=[seq[j] for j in range(len(seq))]
            x=token2id(token_li,token_dict)+[0]*(max_len-len(seq))
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
            x=token2id(token_li,token_dict)+[0]*(max_len-len(seq))
            y=label2id(val_data.label[i],label_dict)
            val_x.append(x)
            val_y.append(y)
        
        val_x=torch.tensor(val_x,dtype=torch.long)
        val_y=torch.tensor(val_y,dtype=torch.long)
        valid=trainset(val_x,val_y) # same dataset build method!
        val_dataloader=DataLoader(valid,batch_size=32,shuffle=False)

        return train_dataloader,val_dataloader
    else:
        test_x=[]
        for i in tqdm(range(data.shape[0])):
            seq=data.sequence[i]
            token_li=[seq[j] for j in range(len(seq))]
            x=token2id(token_li,token_dict)+[0]*(max_len-len(seq))
            test_x.append(x)
        test_x=torch.tensor(test_x,dtype=torch.long)
        test=testset(test_x)
        return DataLoader(test,shuffle=False)
        






        
        
        

        
        


        





