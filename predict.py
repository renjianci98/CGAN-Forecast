import torch
from torch.nn.functional import mse_loss
import pandas as pd
import yaml
import models
from utils.dataloader import ForecastDataset, forecast_dataset_collate
from matplotlib import pyplot as plt
weight_path='checkpoints/weight/2023_03_09_17_22_21/final.pth'
test_path='data/test1.txt'

with open('configs/LSTM.yaml','r',encoding='utf-8') as f:
        configs=yaml.load(f,Loader=yaml.FullLoader)
G=models.make("Generator_LSTM",**configs['Generator'])
a=G.state_dict()
state_dict=torch.load(weight_path,map_location='cuda')
G.load_state_dict(state_dict)
G.eval()
test_dataset=ForecastDataset(test_path,test=True)
total_mse=0
speed_result_list=[]
speed_gt_list=[]
with torch.no_grad():
    for i in range(500):
        history_data,forecast_gt=test_dataset[i]
        history_data=history_data.unsqueeze(0)
        noise=torch.rand((1,128),device='cuda')
        forecast_result=G(history_data,noise)
        forecast_result=forecast_result.squeeze(0)
        speed_result=forecast_result[:,1].tolist()
        speed_gt=forecast_gt[:,1].tolist()
        speed_result_list.extend(speed_result)
        speed_gt_list.extend(speed_gt)
        total_mse+=mse_loss(forecast_result,forecast_gt)
mse=total_mse/len(test_dataset)
plt.figure(figsize=(20,8))
plt.plot(speed_gt_list,color='blue')
plt.plot(speed_result_list,color='red')
plt.show()
plt.savefig('predict_result/1.jpg')
print(mse)