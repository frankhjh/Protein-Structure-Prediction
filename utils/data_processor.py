#!/usr/bin/env python
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict
import re

class train_processor():

    def __init__(self,path):
        self.path=path

    def loader(self):
        with open(self.path,'r') as train_file:
            out=[]
            for line in tqdm(train_file):
                if line.startswith('>'):
                    out.append([])
                out[-1].append(line.strip())
        out=list(map(lambda x:''.join(x),out))
        out=[out[i].split('}') for i in range(len(out))]
        
        dic_out=defaultdict(list)
        for item in out:
            tmp=re.search(r'[a-z]\.[0-9]+\.[0-9]+\.[0-9]+',item[0]).group().split('.')
            label='.'.join(tmp[:2])
            sequence=item[1]
            
            dic_out['label'].append(label)
            dic_out['sequence'].append(sequence)
        
        df=pd.DataFrame(dic_out)
        return df

class test_processor():

    def __init__(self,path):
        self.path=path
    
    def loader(self):
        test_dict=defaultdict(list)
        
        with open(self.path,'r') as test_file:
            tmp=[]
            for line in tqdm(test_file):
                if line.startswith('>'):
                    tmp.append([])
                tmp[-1].append(line.strip())
            
            for item in tmp:
                test_dict['sample_id'].append(item[0][1:])
                test_dict['sequence'].append(''.join(item[1:]))

        df=pd.DataFrame(test_dict)   
        return df
    











        






