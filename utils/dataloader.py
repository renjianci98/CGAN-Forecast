import numpy as np
import torch
import pandas as pd

from torch.utils.data.dataset import Dataset


class ForecastDataset(Dataset):
    def __init__(self, path, test=False):
        self.test = test
        self.df=pd.read_csv('data/normalized_data.csv')
        with open(path,'r',encoding='utf-8') as f:
            ls=f.readlines()
        self.index_ls=[int(s) for s in ls]
        self.length=len(self.index_ls)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        '''
        转换为数据中的index
        '''
        index=self.index_ls[index]
        history = self.df.iloc[index:index+48, 5:]
        predict_gt = self.df.iloc[index+48:index+52, [11,15]]
        history = torch.tensor(history.values, dtype=torch.float)
        predict_gt = torch.tensor(predict_gt.values, dtype=torch.float)
        if self.test:
            history = history.cuda()
            predict_gt = predict_gt.cuda()
        return history, predict_gt
    
def forecast_dataset_collate(batch):
    '''
    DataLoader中collate_fn使用
    '''
    histories = []
    predict_gts = []
    for h, pg in batch:
        histories.append(h)
        predict_gts.append(pg)
    histories = torch.stack(histories, dim=0)
    predict_gts = torch.stack(predict_gts, dim=0)
    return histories, predict_gts
