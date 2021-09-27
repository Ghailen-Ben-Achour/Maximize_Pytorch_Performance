# Maximize_Pytorch_Performance
This repository presents faster and more effective ways to train neural networks.

## Torchscript
```torch.jit``` enables to move from **eager execution** to **graph execution**.

|batch size| Architecture |Module | GPU Training Time | 
|----------|--------------|-------|-----|
|100| CNN | nn.module  | 16.46s  |
|100| CNN | ScriptModule  | 16.8s  |
|8| GAN | nn.module  | 34.62s  |
|8| GAN | ScriptModule  | 28.16s  |

Generally, The effect of ```torch.jit``` is better using GPUs. (on CPUs results are expected to be similar)

## Mixed Precision
Mixed precision allows faster operations. It is a technique for substantially reducing neural net training time
