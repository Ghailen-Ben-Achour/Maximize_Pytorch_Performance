# Maximize_Pytorch_Performance
This repository presents faster and more effective ways to train/deploy neural networks.

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
Mixed precision allows faster operations. It is a technique for substantially reducing neural net training time by performing as many operations as possible in half-precision floating point, FP16, instead of the (PyTorch default) single-precision floating point, FP32. the ```torch.cuda.amp``` mixed-precision training module forthcoming could deliver speed-ups of 50-60% in large model training jobs with just a handful of new lines of code.
To train a basic CNN model with mixed precision, access ```mixed_precision``` folder and run the command below
```bash
python main.py --mixed_prec True
```
