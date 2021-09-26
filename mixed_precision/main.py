import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from modules import CNN_scripted, CNN
from torch import optim
from torch import nn
from argparse import ArgumentParser
import train

def get_args():
    parser = ArgumentParser(description='Torchscript module')
    parser.add_argument('--script_module', type=bool, default=true)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--mixed_prec', type=bool, default=False)
    args = parser.parse_args()
    return args


args = get_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST(
    root = '../../data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = '../../data', 
    train = False, 
    transform = ToTensor()
)

loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=args.batch_size, 
                                          shuffle=True, 
                                          num_workers=2),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=args.batch_size, 
                                          shuffle=True, 
                                          num_workers=2),
}

if (args.sript_module == 'scripted'):
  cnn = CNN_scripted().to(device)
else:
  cnn = CNN().to(device)

optimizer = optim.Adam(cnn.parameters(), lr = 0.01) 
loss_func = nn.CrossEntropyLoss()   
num_epochs = args.epoch

print(args.mixed_prec)
if args.mixed_prec==True:
  print("training mixed precision")
  train.train_mixed_prec(num_epochs, cnn, loaders, loss_func, optimizer, device)
else:
  print("training float32")
  train.train(num_epochs, cnn, loaders, loss_func, optimizer, device)
