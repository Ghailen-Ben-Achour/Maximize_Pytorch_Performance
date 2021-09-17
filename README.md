# Maximize_Pytorch_Performance
Pytorch tips to train NN faster

## Torchscript


|batch size| Architecture |Module | CPU  | GPU | 
|----------|--------------|-------|------|-----|
|100| CNN | nn.module  | 79s |  16.46s  |
|100| CNN | ScriptModule  | 82s |  16.8s  |
|8| GAN | nn.module  | ?? |  34.62s  |
|8| GAN | ScriptModule  | ?? |  28.16s  |
