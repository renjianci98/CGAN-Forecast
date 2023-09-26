import numpy as np
import torch
import pandas as pd

from torch.utils.data.dataset import Dataset


class ForecastDataset(Dataset):
    def __init__(self, configs,ls_path, test=False):
        self.test = test
        self.dataset_name=configs['name']
        self.history_len=configs['history_len'] #历史数据长度
        self.predict_len=configs['predict_len'] #预测数据长度
        self.attr_start=configs['attribute_start'] #属性从哪一列开始
        self.normalize_factor=[]
        df=pd.read_csv(configs['path'])
        self.df=self.normalize(df)
        with open(ls_path,'r',encoding='utf-8') as f:
            ls=f.readlines()
        self.ls_index=[int(s) for s in ls]
        self.length=len(self.ls_index)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        '''
        list中的index转换为数据集中的index
        '''
        index=self.ls_index[index]
        history = self.df.iloc[index:index+self.history_len, self.attr_start:]
        predict_gt = self.df.iloc[index+self.history_len:index+self.history_len+self.predict_len, [11,15]]
        history = torch.tensor(history.values, dtype=torch.float)
        predict_gt = torch.tensor(predict_gt.values, dtype=torch.float)
        if self.test:
            history = history.cuda()
            predict_gt = predict_gt.cuda()
        return history, predict_gt
    def normalize(self,df):
        for col in list(df.columns)[self.attr_start:]:
            Max = np.max(df[col])
            Min = np.min(df[col])
            df[col] = (df[col] - Min)/(Max - Min)
            self.normalize_factor.append((Max-Min,Min))
        return df
    def denormalize(self,data,column):
        return data*self.normalize_factor[column][0]+self.normalize_factor[column][1]


    
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
