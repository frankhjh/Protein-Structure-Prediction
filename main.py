#!/usr/bin/env python
import pandas as pd
import os
import torch
import torch.nn
from torch import optim
from model.mk_cnn import multi_kernel_cnn
from train import evalute,train
from prepare_data import build_token_dict,build_label_dict,max_len,token2id,label2id,prep_dataloader
from utils.data_processor import train_processor,test_processor
import argparse

parser=argparse.ArgumentParser(description='training parameters')
parser.add_argument('--epochs',type=int)
parser.add_argument('--device',type=str)
args=parser.parse_args()

def predict():
    pass

def submit():
    pass

def main():

    # build train_df & test_df
    train_p=train_processor(path='./data/astral_train.fa')
    train_df=train_p.loader()
    test_p=test_processor(path='./data/astral_test.fa')
    test_df=test_p.loader()
    print('>>train/test df built!')

    # token_dict
    token_dict=build_token_dict(train_df)
    # label dict
    label_dict=build_label_dict(train_df)
    # get max len
    maxlen=max_len(train_df,test_df)

    # prepare train_dataloader/val_dataloader & test_dataloader
    train_dataloader,val_dataloader=prep_dataloader('train',train_df,token_dict,label_dict,maxlen)
    test_dataloader=prep_dataloader('test',test_df,token_dict,label_dict,maxlen)
    print('>>dataloader prepared!')

    # load model
    model=multi_kernel_cnn(vocab_size=len(token_dict),max_len=maxlen,padding_idx=0,num_classes=len(label_dict))
    print('>>model loaded!')

    # train 
    epochs=args.epochs
    device=args.device
    train(model,train_dataloader,val_dataloader,epochs,device)

    # predict

    # submit


if __name__=='__main__':
    main()


