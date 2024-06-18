import grn_run_abandoned
from grn_run_abandoned import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from video_model import Transformer
sys.path.append('../WENDY')

device = torch.device("mps") if torch.cuda.is_available() else "cpu"
max_matrix_size = 20

samples = np.load('../WENDY/sampl_225.npy', allow_pickle=True)
True_Grns = []
Wendy_ests = np.load('../WENDY/wendy_gens_225.npy', allow_pickle=True)


for i in range(len(samples[:15])):
    True_Grns.append(samples[i][0].true_grn)


Wendy_ests = Wendy_ests[:15]

model = WENDY_Transformer(
    max_matrix_size=max_matrix_size,
    dim_model=64,
    num_heads=8,
    num_encoder_layers = 3,
    num_decoder_layers= 3,
    dropout_p=0.1
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Wendy_ests = torch.tensor(Wendy_ests).to(device)
# Wendy_ests = Wendy_ests.to(torch.float64)
# True_Grns = torch.tensor(True_Grns).to(device)
# True_Grns = True_Grns.to(torch.float64)

train_dataloader = batchify_data(input_matrices=Wendy_ests[:10], true_matrices=True_Grns[:10], batch_size=1, max_matrix_size=20)
val_dataloader = batchify_data(input_matrices=Wendy_ests[10:], true_matrices=True_Grns[10:], batch_size=1, max_matrix_size=20)
#print(Wendy_ests[0][0])
print((train_dataloader[0][1]))
train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, 10, device=device)
