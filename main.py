import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
from data_utils import download_read_csv, MovieLens
from evaluate import metrics
from MLP import MLP
from GMF import GMF

############################# CONFIGURATION #############################
os.environ['KMP_DUPLICATE_LIB_OK']=''
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

parser = argparse.ArgumentParser(description='Select Parameters')
parser.add_argument('-m', '--model', type=str, default='MLP', help='select among the following model,[MLP, GMF, NeuMF]')
parser.add_argument('-nf', '--num_factors', type=int, default=8, help='number of predictive factors : [8, 16, 32, 64]')
parser.add_argument('-nl', '--num_layers', type=int, default=3, help='number of hidden layers in MLP Model : [0, 1, 2, 3, 4]')
parser.add_argument('-b', '--batch', type=int, default=128, help='batch size : [128, 256, 512, 1024]')
parser.add_argument('-e', '--epoch', type=int, default=10, help='number of epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate : [0.0001, 0.0005, 0.001, 0.005]')
parser.add_argument('-tk', '--top_k', type=int, default=10)
args = parser.parse_args()

############################## PREPARE DATASET ##########################
root_path = "dataset"
file_name = "ml-latest-small"
#file_name = "ml-latest"
file_type = ".zip"

data = download_read_csv(root=root_path, filename=file_name, filetype=file_type, download=True)
train_ratings, test_ratings = data.data_processing()

# 각각 Dataset 객체 할당
train_data = MovieLens(ratings=train_ratings, ng_num=4)
test_data = MovieLens(ratings=test_ratings, ng_num=99)

num_users, num_items = train_data.get_num()

# Datasetloader
train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True, num_workers=0)
test_dataloader = DataLoader(dataset=test_data, batch_size=100, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
if args.model == 'MLP':
    model = MLP(num_users, num_items, args.num_factors, args.num_layers)
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

else:
    model = GMF(num_users, num_items, args.num_factors)
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

########################### TRAINING #####################################
for epoch in range(args.epoch):
  for user, item, label in train_dataloader:
    user = user.to(device)
    item = item.to(device)
    label = label.float().to(device)

    # gradient 초기화
    model.zero_grad()
    prediction = model(user, item)
    loss = loss_function(prediction, label)
    loss.backward()
    optimizer.step()

  HR, NDCG = metrics(model, test_dataloader, args.top_k, device)
  print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
