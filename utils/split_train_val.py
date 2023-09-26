import pandas as pd
import random


def write_file(path, ls):
    ls=[str(i) for i in ls]
    with open(path, 'w', encoding='utf-8') as f:
        for s in ls:
            f.write(f'{s}\n')


df = pd.read_csv('data/original_data.csv')
predict_len=52
total_num = len(df)-predict_len  # 有效风速序列数
train_percent = 0.9
trainval_percent = 0.9
random.seed()
index = range(total_num)
tv_num = int(trainval_percent*total_num)
t_num = int(train_percent*tv_num)
trainval = random.sample(index, tv_num)
test = list(set(index)-set(trainval))
train = random.sample(trainval, t_num)
val = list(set(trainval)-set(train))
write_file('data/trainval.txt',trainval)
write_file('data/train.txt',train)
write_file('data/val.txt',val)
write_file('data/test.txt',test)
test1=list(range(5000))
write_file('data/test1.txt',test1)