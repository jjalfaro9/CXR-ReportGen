'''Global settings.'''

import torch


# batch_size = 32 is too larger for my 1080 TI
batch_size = 24
cuda = True
device = torch.device('cuda')
num_epochs = 30
seed = 2718
