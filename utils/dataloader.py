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
        history_data = self.df.iloc[index:index+48, 5:]
        forecast_gt = self.df.iloc[index+48:index+52, [11,15]]
        history_data = torch.tensor(history_data.values, dtype=torch.float)
        forecast_gt = torch.tensor(forecast_gt.values, dtype=torch.float)
        if self.test:
            history_data = history_data.cuda()
            forecast_gt = forecast_gt.cuda()
        return history_data, forecast_gt
    
def forecast_dataset_collate(batch):
    '''
    DataLoader中collate_fn使用
    '''
    history_datas = []
    forecast_gt = []
    for hd, fg in batch:
        history_datas.append(hd)
        forecast_gt.append(fg)
    history_datas = torch.stack(history_datas, dim=0)
    forecast_gt = torch.stack(forecast_gt, dim=0)
    return history_datas, forecast_gt
