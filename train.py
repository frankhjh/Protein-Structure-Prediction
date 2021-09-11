#!/usr/bin/env python
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim


def evalute(metric,model,loader,device):
    val_loss=0.0
    for step,(x,y) in enumerate(loader):
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            output=model(x)
            loss=metric(output,y)
            val_loss+=loss.item()
    return val_loss/(step+1)
        
        
def train(model,train_dataloader,class_weights,val_dataloader,epochs,device,lr=1e-3):
    m=model
    optimizer=optim.Adam(m.parameters(),lr=lr)
    metric=nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    min_loss,best_epoch=10000.0,1
    print('>>Start Training...')
    for epoch in range(1,epochs+1):
        total_loss=0.0
        for step,(x,y) in tqdm(enumerate(train_dataloader)):
            x,y=x.to(device),y.to(device)
            output=m(x)
            loss=metric(output,y)
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            total_loss+=loss.item()
        avg_loss=total_loss/(step+1)

        val_loss=evalute(metric,m,val_dataloader,device)
        if val_loss<min_loss:
            min_loss=val_loss
            best_epoch=epoch
            torch.save(model.state_dict(),'./train_out/bm.ckpt')
        
        print('epoch {},training loss:{}'.format(epoch,avg_loss)+' validation loss:{}'.format(val_loss))
    print('>>Training done!')