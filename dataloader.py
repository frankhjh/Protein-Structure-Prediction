#!/usr/bin/env python
import pandas as pd
import tqdm
import os

class train_process():

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
        return out

    def transform2df(self):
        out=self.loader()
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










        






