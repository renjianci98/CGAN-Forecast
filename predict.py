import torch
from torch.nn.functional import mse_loss
import pandas as pd
import yaml
import models
from utils.dataloader import ForecastDataset, forecast_dataset_collate
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import random
weight_path='checkpoints/weight/2023_09_26_16_50_28/ep620-loss0.401-val_loss0.402.pth'
model_configs_path='configs/LSTM.yaml'
dataset_configs_path='configs/dataset.yaml'
with open(model_configs_path,'r',encoding='utf-8') as f:
        model_configs=yaml.load(f,Loader=yaml.FullLoader)
with open(dataset_configs_path,'r',encoding='utf-8') as f:
        dataset_configs=yaml.load(f,Loader=yaml.FullLoader)
dataset_name=dataset_configs['name']
predict_len=dataset_configs['predict_len']
G=models.make("Generator_LSTM",**model_configs['Generator'])
a=G.state_dict()
state_dict=torch.load(weight_path,map_location='cuda')
G.load_state_dict(state_dict)
G.eval()
test_dataset=ForecastDataset(dataset_configs,f'data/{dataset_name}/test.txt',test=True)
test_len=len(test_dataset)
total_mse=0
speed_result_list=[]
speed_gt_list=[]
sample_numbers=10
random.seed()
with torch.no_grad():
    with tqdm(total=math.ceil(test_len/predict_len),desc='Data predicted',mininterval=0.3) as pbar:
        for i in range(0,test_len,predict_len):
                history,predict_gt=test_dataset[i]
                history=history.unsqueeze(0)
                for j in range(sample_numbers):
                        noise=torch.rand((1,128),device='cuda')
                        sample_result=G(history,noise)
                        sample_result=sample_result.squeeze(0)
                        if j==0:
                                predict_result=torch.zeros_like(sample_result)
                        predict_result+=sample_result
                predict_result/=sample_numbers
                speed_result=test_dataset.denormalize(predict_result[:,1],10)
                speed_gt=test_dataset.denormalize(predict_gt[:,1],10)
                total_mse+=mse_loss(speed_result,speed_gt)
                speed_result_list.extend(speed_result.tolist())
                speed_gt_list.extend(speed_gt.tolist())   
                pbar.update(1)   
mse=total_mse/(math.ceil(test_len/predict_len))
plt.figure(figsize=(20,8))
plt.plot(speed_gt_list[0:1000],color='blue',label='gt')
plt.plot(speed_result_list[0:1000],color='red',label='result')
plt.legend()
plt.show()
plt.savefig('predict_result/result.jpg')
print(mse)

