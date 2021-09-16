import math
import torch
import torch.nn as nn
from modules import Generator, Scripted_Generator, Discriminator, Scripted_Discriminator
from train import generate_even_data, train
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Torchscript module')
    parser.add_argument('--module', type=str, default='scripted')
    parser.add_argument('--batch_size', type=int, default=8)
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
if args.module == 'scripted':
  generator = Scripted_Generator(input_length).to(device)
  discriminator = Scripted_Discriminator(input_length).to(device)
else:
  generator = Generator(input_length).to(device)
  discriminator = Discriminator(input_length).to(device)

# Optimizers
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# loss
loss = nn.BCELoss()
train(training_steps, generator, discriminator, loss, generator_optimizer,
                   discriminator_optimizer, batch_size, input_length, max_int, device)
