import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # border1s = [0,            12 * 30 * 24 - self.seq_len,     12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24,      12 * 30 * 24 + 8 * 30 * 24]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # print("Triggered!!!!!")
        dataset_len = len(df_raw)
        border1s = [0, int(dataset_len*0.8),int(dataset_len*0.9)]
        border2s = [int(dataset_len*0.8),int(dataset_len*0.9),len((df_raw))]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_btc(Dataset):
    def __init__(self, args=[], root_path="dataset/btc", flag='train', size=None,
                 features='MS', data_path='btc_train.csv',
                 target='range5', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # print("Yes right dataset is loaded")
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 0
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # print(f"scale {scale}, seq_len {self.seq_len},  self.label_len {self.label_len}, self.pred_len {self.pred_len}")

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # border1s = [0,            12 * 30 * 24 - self.seq_len,     12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24,      12 * 30 * 24 + 8 * 30 * 24]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # print("Triggered!!!!!")
       
        dataset_len = len(df_raw)
        border1s = [0,                       int(dataset_len*0.8*0.5), int(dataset_len*0.5)]
        border2s = [int(dataset_len*0.8*0.5),int(dataset_len*0.5),     int(dataset_len)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[2:]
            # print(cols_data)
            df_data = df_raw[cols_data]


        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.time.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 




        # self.data_x = data[border1:border2][:,1:]

        # self.data_y = data[border1:border2][:,:1]

        self.data_x = data[border1:border2]

        self.data_y = data[border1:border2][:,:1]
        # print(self.data_x.shape,self.data_y.shape)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        r_begin = s_end 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.label_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class mDataset_btc_block(Dataset):
    def __init__(self, args=[], root_path="dataset/btc", flag='train', size=None,
                 features='MS', data_path='btc_t_v_withf.csv',
                 target='range5', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # print("Yes right dataset is loaded")
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 0
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        print(f"data: {data_path}, scale {scale}, seq_len {self.seq_len},  self.label_len {self.label_len}, self.pred_len {self.pred_len}")

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # border1s = [0,            12 * 30 * 24 - self.seq_len,     12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24,      12 * 30 * 24 + 8 * 30 * 24]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # print("Triggered!!!!!")
       
        dataset_len = len(df_raw)
        border1s = [0,                       int(dataset_len*0.8*0.5), int(dataset_len*0.5)]
        border2s = [int(dataset_len*0.8*0.5),int(dataset_len*0.5),     int(dataset_len)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[2:]
            # print(cols_data)
            df_data = df_raw[cols_data]


        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.time.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 




        self.data_x = data[border1:border2][:,1:]

        self.data_y = data[border1:border2][:,:1]


        # self.data_x = data[border1:border2]

        # self.data_y = data[border1:border2][:,:1]
        # print(self.data_x.shape,self.data_y.shape)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # s_begin = index
        s_begin = index*self.seq_len
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        r_begin = s_end - 1
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.label_len + 1
        return len(self.data_x)//self.seq_len
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class m4Dataset_btc_block(Dataset):
    def __init__(self, args=[], root_path="dataset/btc", flag='train', size=None,
                 features='MS', data_path='btc_.csv',
                 target='range5', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # print("Yes right dataset is loaded")
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 0
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        print(f"data: {data_path}, scale {scale}, seq_len {self.seq_len},  self.label_len {self.label_len}, self.pred_len {self.pred_len}")

        # print(f"scale {scale}, seq_len {self.seq_len},  self.label_len {self.label_len}, self.pred_len {self.pred_len}")

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # border1s = [0,            12 * 30 * 24 - self.seq_len,     12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24,      12 * 30 * 24 + 8 * 30 * 24]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # print("Triggered!!!!!")
       
        dataset_len = len(df_raw)
        border1s = [0,                       int(dataset_len*0.8*0.8), int(dataset_len*0.8)]
        border2s = [int(dataset_len*0.8*0.8),int(dataset_len*0.8),     int(dataset_len)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[2:]
            # print(cols_data)
            df_data = df_raw[cols_data]


        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.time.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 




        self.data_x = data[border1:border2][:,1:]

        self.data_y = data[border1:border2][:,:1]

        # self.data_x = data[border1:border2]

        # self.data_y = data[border1:border2][:,:1]
        # print(self.data_x.shape,self.data_y.shape)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index*self.seq_len
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        r_begin = s_end - 1
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.label_len + 1
        return len(self.data_x)//self.seq_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class mDataset_btc_CGNN(Dataset):
    def __init__(self, args=[], root_path="dataset/btc", flag='train', size=None,
                 features='MS', data_path='btc_t_v_withf.csv',
                 target='range5', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # print("Yes right dataset is loaded")
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 0
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # print(f"data: {data_path}, scale {scale}, seq_len {self.seq_len},  self.label_len {self.label_len}, self.pred_len {self.pred_len}")

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cache_path = "dataset/graph/CGNN_edges"

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        if self.args.GNN_type != 1 and flag=='train': # static and hybird all require a global graph
            # print(f"shape of data_x {self.data_x.shape}")
            self.edge_index, self.edge_attr = mDataset_btc_CGNN.create_graph(self.data_x)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
       
        dataset_len = len(df_raw)
        border1s = [0,                       int(dataset_len*0.8*0.5), int(dataset_len*0.5)]
        border2s = [int(dataset_len*0.8*0.5),int(dataset_len*0.5),     int(dataset_len)]        
        # border1s = [0,                       int(dataset_len*0.8*0.75), int(dataset_len*0.75)]
        # border2s = [int(dataset_len*0.8*0.75),int(dataset_len*0.75),     int(dataset_len)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[2:]
            # print(cols_data)
            df_data = df_raw[cols_data]


        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.time.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 




        self.data_x = data[border1:border2][:,1:]

        self.data_y = data[border1:border2][:,:1]


        # self.data_x = data[border1:border2]

        # self.data_y = data[border1:border2][:,:1]
        # print(self.data_x.shape,self.data_y.shape)
        self.data_stamp = data_stamp

    @staticmethod
    def create_graph(input, threshold=0.5):


        if len(input.shape) == 2:
            num_features, num_nodes = input.shape
        elif len(input.shape) == 3:
            batch_size, num_features, num_nodes= input.shape

        flattened_data = input.reshape(-1,num_nodes)
        # print(f"shape of graph input {flattened_data.shape}")

        corr_matrix = np.corrcoef(flattened_data.T)
        edge_index = []
        edge_attr = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and abs(corr_matrix[i, j]) > threshold:
                    edge_index.append([i, j])
                    edge_attr.append(abs(corr_matrix[i, j]))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Reshape to (num_edges, 1)
        # print(f"edge shape from graph {edge_index.shape} \n {edge_index}\n-----\nedge attr {edge_attr.shape} \n {edge_attr}")
        return edge_index, edge_attr
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        r_begin = s_end - 1
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]


        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.label_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class m4Dataset_btc_CGNN(Dataset):
    def __init__(self, args=[], root_path="dataset/btc", flag='train', size=None,
                 features='MS', data_path='btc_t_v_withf.csv',
                 target='range5', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # print("Yes right dataset is loaded")
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 0
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # print(f"data: {data_path}, scale {scale}, seq_len {self.seq_len},  self.label_len {self.label_len}, self.pred_len {self.pred_len}")

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cache_path = "dataset/graph/CGNN_edges"

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        if self.args.GNN_type != 1 and flag=='train': # static and hybird all require a global graph
            # print(f"shape of data_x {self.data_x.shape}")
            self.edge_index, self.edge_attr = mDataset_btc_CGNN.create_graph(self.data_x)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
       
        dataset_len = len(df_raw)
        # border1s = [0,                       int(dataset_len*0.8*0.5), int(dataset_len*0.5)]
        # border2s = [int(dataset_len*0.8*0.5),int(dataset_len*0.5),     int(dataset_len)]        
        border1s = [0,                       int(dataset_len*0.8*0.8), int(dataset_len*0.8)]
        border2s = [int(dataset_len*0.8*0.8),int(dataset_len*0.8),     int(dataset_len)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[2:]
            # print(cols_data)
            df_data = df_raw[cols_data]


        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.time.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 




        self.data_x = data[border1:border2][:,1:]

        self.data_y = data[border1:border2][:,:1]


        # self.data_x = data[border1:border2]

        # self.data_y = data[border1:border2][:,:1]
        # print(self.data_x.shape,self.data_y.shape)
        self.data_stamp = data_stamp

    @staticmethod
    def create_graph(input, threshold=0.5):


        if len(input.shape) == 2:
            num_features, num_nodes = input.shape
        elif len(input.shape) == 3:
            batch_size, num_features, num_nodes= input.shape

        flattened_data = input.reshape(-1,num_nodes)
        # print(f"shape of graph input {flattened_data.shape}")

        corr_matrix = np.corrcoef(flattened_data.T)
        edge_index = []
        edge_attr = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and abs(corr_matrix[i, j]) > threshold:
                    edge_index.append([i, j])
                    edge_attr.append(abs(corr_matrix[i, j]))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Reshape to (num_edges, 1)
        # print(f"edge shape from graph {edge_index.shape} \n {edge_index}\n-----\nedge attr {edge_attr.shape} \n {edge_attr}")
        return edge_index, edge_attr
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        r_begin = s_end - 1
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]


        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.label_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class mDataset_btc(Dataset):
    def __init__(self, args=[], root_path="dataset/btc", flag='train', size=None,
                 features='MS', data_path='btc_t_v_withf.csv',
                 target='range5', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # print("Yes right dataset is loaded")
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 0
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        print(f"data: {data_path}, scale {scale}, seq_len {self.seq_len},  self.label_len {self.label_len}, self.pred_len {self.pred_len}")

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # border1s = [0,            12 * 30 * 24 - self.seq_len,     12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24,      12 * 30 * 24 + 8 * 30 * 24]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # print("Triggered!!!!!")
       
        dataset_len = len(df_raw)
        border1s = [0,                       int(dataset_len*0.8*0.5), int(dataset_len*0.5)]
        border2s = [int(dataset_len*0.8*0.5),int(dataset_len*0.5),     int(dataset_len)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[2:]
            # print(cols_data)
            df_data = df_raw[cols_data]


        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.time.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 




        self.data_x = data[border1:border2][:,1:]

        self.data_y = data[border1:border2][:,:1]


        # self.data_x = data[border1:border2]

        # self.data_y = data[border1:border2][:,:1]
        # print(self.data_x.shape,self.data_y.shape)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # s_begin = index
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        r_begin = s_end - 1
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.label_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class m4Dataset_btc(Dataset):
    def __init__(self, args=[], root_path="dataset/btc", flag='train', size=None,
                 features='MS', data_path='btc_.csv',
                 target='range5', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # print("Yes right dataset is loaded")
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 0
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        print(f"data: {data_path}, scale {scale}, seq_len {self.seq_len},  self.label_len {self.label_len}, self.pred_len {self.pred_len}")

        # print(f"scale {scale}, seq_len {self.seq_len},  self.label_len {self.label_len}, self.pred_len {self.pred_len}")

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # border1s = [0,            12 * 30 * 24 - self.seq_len,     12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24,      12 * 30 * 24 + 8 * 30 * 24]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # print("Triggered!!!!!")
       
        dataset_len = len(df_raw)
        border1s = [0,                       int(dataset_len*0.8*0.8), int(dataset_len*0.8)]
        border2s = [int(dataset_len*0.8*0.8),int(dataset_len*0.8),     int(dataset_len)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[2:]
            # print(cols_data)
            df_data = df_raw[cols_data]


        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.time.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 




        self.data_x = data[border1:border2][:,1:]

        self.data_y = data[border1:border2][:,:1]

        # self.data_x = data[border1:border2]

        # self.data_y = data[border1:border2][:,:1]
        # print(self.data_x.shape,self.data_y.shape)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        r_begin = s_end - 1
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.label_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


