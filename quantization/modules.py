from torch import nn
import torch

class Generator(nn.Module):

    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()
    def forward(self, x):
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.activation(x)
      return x

class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense1 = nn.Linear(int(input_length), int(input_length))
        self.dense2 = nn.Linear(int(input_length), 1)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.dense1(x)
        x = self.dense1(x)
        x = self.dense1(x)
        x = self.dense1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.activation(x)
        return x

class Scripted_Generator(torch.jit.ScriptModule):

    def __init__(self, input_length: int):
        super(Scripted_Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()
    @torch.jit.script_method
    def forward(self, x):
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.dense_layer(x)
      x = self.activation(x)
      return x

class Scripted_Discriminator(torch.jit.ScriptModule):
    def __init__(self, input_length: int):
        super(Scripted_Discriminator, self).__init__()
        self.dense1 = nn.Linear(int(input_length), int(input_length))
        self.dense2 = nn.Linear(int(input_length), 1)
        self.activation = nn.Sigmoid()
    @torch.jit.script_method
    def forward(self, x):
        x = self.dense1(x)
        x = self.dense1(x)
        x = self.dense1(x)
        x = self.dense1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.activation(x)
        return x

