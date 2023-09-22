import models
import torch
import yaml
with open('configs/transformer.yaml','r',encoding='utf-8') as f:
    configs=yaml.load(f,Loader=yaml.FullLoader)
input=torch.rand((32,96,14),device='cuda')
Z=torch.rand((96,32,128),device='cuda')
gen=models.make('Generator_Transformer',**configs['Generator'])
gen=gen.cuda()
output=gen(input,Z)
print(output.shape)