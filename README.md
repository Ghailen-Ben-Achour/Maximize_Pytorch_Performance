# Maximize_Pytorch_Performance
Pytorch tips to train NN faster

## Torchscript
```torch.jit``` enables to move from **eager execution** to **graph execution**.

|batch size| Architecture |Module | GPU Training Time | 
|----------|--------------|-------|-----|
|100| CNN | nn.module  | 16.46s  |
|100| CNN | ScriptModule  | 16.8s  |
|8| GAN | nn.module  | 34.62s  |
|8| GAN | ScriptModule  | 28.16s  |
