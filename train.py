import os
import datetime
import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
import models
from utils.fit_epoch import fit_one_epoch
from utils.callbacks import LossHistory
from utils.dataloader import ForecastDataset, forecast_dataset_collate



if __name__ == "__main__":
    '''
    训练参数设置
    '''
    train_gpu = [0, 1, 2, 3]
    Init_Epoch = 0
    Final_Epoch = 1000
    save_period = 10
    batch_size = 1024
    D_Init_lr = 1e-4
    G_Init_lr = 1e-4
    D_steps = 1
    G_steps = 1
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    weight_save_dir = 'checkpoints/weight/{}'.format(time_str)
    log_save_dir='checkpoints/log'
    num_workers = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    train_path = 'data/train.txt'
    val_path = 'data/val.txt'
    print('Number of devices: {}'.format(ngpus_per_node))

    with open('configs/LSTM.yaml','r',encoding='utf-8') as f:
        configs=yaml.load(f,Loader=yaml.FullLoader)

    '''
    保存log以及权重的路径初始化
    '''
    if not os.path.exists(weight_save_dir):
        os.makedirs(weight_save_dir)
    G_loss_history = LossHistory(log_save_dir, 'G')
    D_loss_history = LossHistory(log_save_dir, 'D')

    '''
    根据配置生成模型
    '''
    G = models.make('Generator_LSTM',**configs['Generator']).train()
    D = models.make('Discriminator_LSTM',**configs['Discriminator']).train()
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)
    cudnn.benchmark = True
    G = G.cuda()
    D = D.cuda()

    '''
    准备数据
    '''
    train_dataset = ForecastDataset(train_path)
    val_dataset = ForecastDataset(val_path)
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=forecast_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=forecast_dataset_collate)

    '''
    损失函数、优化器和学习率调整
    '''
    loss_fn = torch.nn.MSELoss()
    G_optimizer = {
        'adam': optim.Adam(G.parameters(), G_Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(G.parameters(), G_Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    D_optimizer = {
        'adam': optim.Adam(D.parameters(), D_Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(D.parameters(), D_Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]
    G_lr_scheduler = optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=0.99)
    D_lr_scheduler = optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.99)


    for epoch in range(Init_Epoch, Final_Epoch):
        D_steps,G_steps=fit_one_epoch(G, D, G_optimizer, D_optimizer,G_steps, D_steps, loss_fn, batch_size, epoch, epoch_step, epoch_step_val,
                      gen, gen_val, Final_Epoch, save_period, weight_save_dir, G_loss_history, D_loss_history)
        G_lr_scheduler.step()
        D_lr_scheduler.step()
    G_loss_history.writer.close()
    D_loss_history.writer.close()
