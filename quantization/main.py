import math
import torch
import torch.nn as nn
from modules import Generator, Scripted_Generator, Discriminator, Scripted_Discriminator
from argparse import ArgumentParser
from time import time

def get_args():
    parser = ArgumentParser(description='Torchscript module')
    parser.add_argument('--module_script', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=2500)
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--input_length', type=int, default=128)
    args = parser.parse_args()
    return args

args = get_args()
max_int= args.input_length
batch_size= args.batch_size
training_steps = args.epoch

input_length = int(math.log(max_int, 2))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Models
if args.module_script:
  generator = Scripted_Generator(input_length)
  discriminator = Scripted_Discriminator(input_length)
else:
  generator = Generator(input_length)
  discriminator = Discriminator(input_length)

quant_gen = torch.quantization.quantize_dynamic(
     generator,  # the original model
     {nn.Linear},  # a set of layers to dynamically quantize
     dtype=torch.qint8)  # the target dtype for quantized weights
     
quant_dis = torch.quantization.quantize_dynamic(
     discriminator,  # the original model
     {nn.Linear},  # a set of layers to dynamically quantize
     dtype=torch.qint8)  # the target dtype for quantized weights

noise = torch.randint(0, 2, size=(batch_size, input_length)).float()

t0 = time()
discriminator(generator(noise))
print(time()-t0)

t0 = time()
quant_dis(quant_gen(noise))
print(time()-t0)


