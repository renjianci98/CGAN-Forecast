import pandas as pd
import numpy as np
data_frame=pd.read_csv('data/original_data.csv')
print(data_frame.shape)
with open('data/scale_factors.txt','w',encoding='utf-8') as f:
    for col in list(data_frame.columns)[5:]:
        Max = np.max(data_frame[col])
        Min = np.min(data_frame[col])
        data_frame[col] = (data_frame[col] - Min)/(Max - Min)
        scale_factor=(Max-Min,Min)
        f.write(f'{Max-Min} {Min}\n')
data_frame.to_csv('data/normalized_data.csv',index=False)