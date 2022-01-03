import requests
import os
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class download_read_csv():
    def __init__(self, root, filename, filetype, download):
        self.root = root
        self.filename = filename
        self.filetype = filetype
        self.download = download
        if self.download:
            self.download_movielens()
            self.ratings = self.read_ratings_csv()
        else:
            self.ratings = self.read_ratings_csv()

    def download_movielens(self) -> None:  # "-> None" 리턴값을 나타내는 것
        root_path = self.root  # 'dataset'
        file_name = self.filename
        file_type = self.filetype
        url = "http://files.grouplens.org/datasets/movielens/" + file_name + file_type

        # Downloading the file by sending the request to the URL
        req = requests.get(url)

        # 최초 directory 생성
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        with open(os.path.join(root_path, file_name + file_type), 'wb') as output_file:
            output_file.write(req.content)

        # extracting the zip file contents
        zipfile = ZipFile(BytesIO(req.content))
        zipfile.extractall(path=root_path)

        print('Dataset Download Complete')

    def read_ratings_csv(self):
        ratings = pd.read_csv(self.root + '/' + self.filename + '/' + 'ratings.csv')
        ratings = ratings.drop("timestamp", axis=1)
        ratings = sklearn.utils.shuffle(ratings)
        print("ratings.csv Read Complete")
        return ratings

    def data_processing(self):
        train_ratings = self.ratings.copy()
        test_ratings = pd.DataFrame()

        for i in self.ratings['userId'].unique().tolist():
            for j in self.ratings['userId'].index:
                if (self.ratings.iloc[j, 0] == i):
                    test_ratings = pd.concat([test_ratings, pd.DataFrame(self.ratings.iloc[j, :]).T], axis=0)
                    train_ratings = train_ratings.drop(j)
                    break
        # explicit feedback -> implicit feedback
        train_ratings.loc[:, 'rating'] = 1
        test_ratings.loc[:, 'rating'] = 1

        train_ratings = train_ratings.astype(int)
        test_ratings = test_ratings.astype(int)

        '''wrong train test split
        x = self.ratings.copy()
        y = self.ratings['userId']

        # stratified sampling
        train_ratings, test_ratings, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
        '''

        return train_ratings, test_ratings


class MovieLens(Dataset):
    def __init__(self, ratings, ng_num):
        super(MovieLens, self).__init__()
        self.ratings = ratings
        self.ng_num = ng_num
        self.num_users, self.num_items = self.get_num()
        self.all_movieIds = self.get_allmovieIds()
        self.users, self.items, self.labels = self.negative_feedback_augmentation()

    def __len__(self):
        return len(self.users)

    # index에 맞는 sample을 return
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.labels[index]

    def get_num(self):
        num_users = self.ratings['userId'].max() + 1
        num_items = self.ratings['movieId'].max() + 1
        return num_users, num_items

    def get_allmovieIds(self):
        all_movieIds = self.ratings['movieId'].unique()
        return all_movieIds

    def negative_feedback_augmentation(self):
        '''
        ratings.csv는 explicit feedback이다. NCF 논문에 따라 0을 negative feedback으로 가정한다.
        rating column에 존재하는 value는 1로 치환한다.
        '''
        users, items, labels = [], [], []
        user_item_set = set(zip(self.ratings['userId'], self.ratings['movieId']))

        # negative feedback dataset 증가 비율
        negative_ratio = self.ng_num
        for u, i in user_item_set:
            # positive instance
            users.append(u)
            items.append(i)
            labels.append(1)
            # negative instance
            for i in range(negative_ratio):
                # first item random choice
                negative_item = np.random.choice(self.all_movieIds)
                # 해당 item이 user와 interaction이 있었는지 확인하고, interaction이 있었다면 negative_item을 계속 랜덤하게 할당
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(self.all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)