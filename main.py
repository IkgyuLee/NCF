import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from data_utils import Download_read_csv, MovieLens
from evaluate import metrics
from MLP import MLP
from GMF import GMF
from NeuMF import NeuMF

############################# CONFIGURATION #############################
os.environ['KMP_DUPLICATE_LIB_OK']=''
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

parser = argparse.ArgumentParser(description='Select Parameters')
parser.add_argument('-m', '--model_name', type=str, default='MLP', help='select among the following model: [MLP, GMF, NeuMF, NeuMF_pre]')
parser.add_argument('-nf', '--num_factors', type=int, default=8, help='number of predictive factors: [8, 16, 32, 64]')
parser.add_argument('-nl', '--num_layers', type=int, default=3, help='number of hidden layers in MLP Model: [0, 1, 2, 3, 4]')
parser.add_argument('-b', '--batch', type=int, default=128, help='batch size: [128, 256, 512, 1024]')
parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate: [0.0001, 0.0005, 0.001, 0.005]')
parser.add_argument('-tk', '--top_k', type=int, default=10)
parser.add_argument('-pr', '--use_pretrain', type=str, default='False', help='use pretrained model or not')
parser.add_argument('-save', '--save_model', type=str, default='False', help='save trained model or not')
args = parser.parse_args()

print('Model Name: ', format(args.model_name))

# argparse doesn't supprot boolean type
use_pretrain = True if args.use_pretrain =='True' else False
save_model = True if args.save_model == 'True' else False

pretrain_dir = 'pretrain'
if not os.path.exists(pretrain_dir):
    os.makedirs(pretrain_dir)

############################## PREPARE DATASET ##########################
root_path = "dataset"
file_name = "ml-latest-small"
file_type = ".zip"

data = Download_read_csv(root=root_path, filename=file_name, filetype=file_type, download=True)
total_ratings = data.read_ratings_csv()
print("ratings.csv Read Complete")
train_ratings, test_ratings = data.data_processing()

# 각각 Dataset 객체 할당
train_data = MovieLens(total_ratings=total_ratings, ratings=train_ratings, ng_num=4)
test_data = MovieLens(total_ratings=total_ratings, ratings=test_ratings, ng_num=99)

# user, item의 unique 개수 불러오기
num_users, num_items = train_data.get_num()

# Datasetloader
train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True, num_workers=4)
test_dataloader = DataLoader(dataset=test_data, batch_size=100, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
if args.model_name == 'NeuMF':
    neumf = True
else:
    neumf = False

if args.model_name == 'MLP':
    model = MLP(num_users, num_items, args.num_factors, args.num_layers, use_pretrain, neumf)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

elif args.model_name == 'GMF':
    model = GMF(num_users, num_items, args.num_factors, use_pretrain, neumf)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

else: # moel_name = 'NeuMF'
    if use_pretrain:
        GMF_dir = os.path.join(pretrain_dir, 'GMF.pth')
        MLP_dir = os.path.join(pretrain_dir, 'MLP.pth')
        pretrained_GMF = torch.load(GMF_dir)
        pretrained_MLP = torch.load(MLP_dir)

        # 신경망의 모든 매개변수를 고정합니다
        for param in pretrained_GMF.parameters():
            param.requires_grad = False

        for param in pretrained_MLP.parameters():
            param.requires_grad = False

    else:
        pretrained_GMF = None
        pretrained_MLP = None

    model = NeuMF(num_users, num_items, args.num_factors, args.num_layers, neumf, use_pretrain, pretrained_GMF, pretrained_MLP)
    if not use_pretrain:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

model.to(device)
# loss function is Binary Cross Entropy Loss
loss_function = nn.BCELoss()

########################### TRAINING #####################################
start = time.time()
for epoch in range(args.epochs):
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
    print("epoch: {}\tHR: {:.3f}\tNDCG: {:.3f}".format(epoch+1, np.mean(HR), np.mean(NDCG)))

if save_model:
    pretrain_model_dir = os.path.join(pretrain_dir, args.model+'.pth')
    torch.save(model, pretrain_model_dir)

end = time.time()
print(f'Training Time: {end-start:.5f}')