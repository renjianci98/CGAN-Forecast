import pandas as pd
import numpy as np
import yaml
import random
import os

def write_file(path, ls):
    ls=[str(i) for i in ls]
    with open(path, 'w', encoding='utf-8') as f:
        for s in ls:
            f.write(f'{s}\n')

dataset_config='configs/dataset.yaml'
with open(dataset_config,'r',encoding='utf-8') as f:
        configs=yaml.load(f,Loader=yaml.FullLoader)
dataset_name=configs['name']
dataset_path=configs['path']
predict_len=configs['predict_len']
data_frame=pd.read_csv(dataset_path)
total_num = len(data_frame)-configs['history_len']-configs['predict_len']  # 有效风速序列数
train_percent = 0.9
trainval_percent = 0.9
random.seed()
index = range(total_num)
test_num = int((1-trainval_percent)*total_num)
test_start=random.choice(range(total_num-predict_len*test_num))
test=list(range(test_start,test_start+test_num))
trainval = list(set(index)-set(test))
train_num=int(total_num*trainval_percent*train_percent)
train = random.sample(trainval, train_num)
val = list(set(trainval)-set(train))
if not os.path.exists(f'data/{dataset_name}'):
     os.makedirs(f'data/{dataset_name}')
write_file(f'data/{dataset_name}/trainval.txt',trainval)
write_file(f'data/{dataset_name}/train.txt',train)
write_file(f'data/{dataset_name}/val.txt',val)
write_file(f'data/{dataset_name}/test.txt',test)
