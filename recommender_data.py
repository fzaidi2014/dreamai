import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import *
import io
import os
import math
import copy
import pickle
import zipfile
from pathlib import Path

from urllib.error import URLError
from urllib.request import urlopen
from dreamai.utils import *

class MovieLens(Dataset):

    def __init__(self,
                 url_path=None,
                 data_path='movielens',
                 folder_name='ml-latest-small',
                 download=False,
                 isClassifier=False):
       
        if download:
            archive_name = url_path.split('/')[-1]
            self.folder_name, _ = os.path.splitext(archive_name)
            print('downloading archive:{} to folder: {}'.format(url_path,data_path+'/'+self.folder_name))
            try:
                data_obj = urlopen(url_path)
            except URLError as e:
                print('Cannot download the data. Error: %s' % s)
                return 

            assert data_obj.status == 200
            data = data_obj.read()

            with zipfile.ZipFile(io.BytesIO(data)) as arch:
                arch.extractall(data_path)

            print('The archive is extracted into folder: %s' % data_path)
                #print(self.data_len)
        
        else:
            self.folder_name = folder_name

        self.isClassifier = isClassifier

        self.df_ratings,self.df_movies = self.read_data(Path.cwd()/data_path/self.folder_name) 
        #files = self.read_data(Path.cwd()/data_path/self.folder_name) 
        #print(df_ratings.head())
        #print(df_movies.head())
        self.create_dataset()
        
    def read_data(self,path):
        files = {}
        print(path)
        for filename in path.glob('*'):
            print(filename)
            if filename.suffix == '.csv':
                files[filename.stem] = pd.read_csv(filename)
            elif filename.suffix == '.dat':
                if filename.stem == 'ratings':
                    columns = ['userId', 'movieId', 'rating', 'timestamp']
                else:
                    columns = ['movieId', 'title', 'genres']
                data = pd.read_csv(filename, sep='::', names=columns, engine='python')
                files[filename.stem] = data
                #print(files.keys())
        return files['ratings'], files['movies']
        

    def create_dataset(self):
        
        unique_users = self.df_ratings['userId'].unique()
        self.user_to_index = {old: new for new, old in enumerate(unique_users)}
        new_users = self.df_ratings['userId'].map(self.user_to_index)

        unique_movies = self.df_ratings['movieId'].unique()
        self.movie_to_index = {old: new for new, old in enumerate(unique_movies)}
        new_movies = self.df_ratings['movieId'].map(self.movie_to_index)

        self.n_users = unique_users.shape[0]
        self.n_movies = unique_movies.shape[0]

        self.X = pd.DataFrame({'user_id': new_users, 'movie_id': new_movies,
                               'timestamp':self.df_ratings['timestamp']})
        
        if self.isClassifier:
            self.y = self.df_ratings['rating'].astype(np.int64)
            self.y = np.array([x-1 for x in self.y])
        else:
            self.y = self.df_ratings['rating'].astype(np.float32)
            self.y_norm = np.expand_dims(normalize_minmax(self.y),axis=1)
        
        print('unique users = {}, unique movies = {}'.format(self.n_users,self.n_movies))
        
    def __getitem__(self, index):
        x = self.X.iloc[index].values
        if self.isClassifier:
            y = self.y[index]
        else:
            y = self.y_norm[index]
        return (x,y)

    def __len__(self):
        return self.X.shape[0]