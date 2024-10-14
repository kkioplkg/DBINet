import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split




def split(data, repeat):
    # 使用KFold进行5折交叉验证
    kf = KFold(n_splits=5, random_state=repeat, shuffle=True)
    seed = 42
    i = 1
    for train_index, test_index in kf.split(data):
        data_train_val = data.iloc[train_index]
        data_test = data.iloc[test_index]
        data_train, data_val = train_test_split(data_train_val, test_size=0.25, random_state=seed)
        all_index = np.concatenate((data_train.index.values, data_val.index.values, data_test.index.values), axis=0)
        all_index = np.unique(all_index)

        data_train = data_train.reset_index(drop=True)
        data_val = data_val.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)
        path_train = f'./data/repeat{repeat}_fold{i}_train.csv'
        path_val = f'./data/repeat{repeat}_fold{i}_val.csv'
        path_test = f'./data/repeat{repeat}_fold{i}_test.csv'
        edge_index_path = f'./data/repeat{repeat}_fold{i}_edge_index.npy'

        os.makedirs(os.path.dirname(path_train), exist_ok=True)

        data_train.to_csv(path_train, index=False)
        data_val.to_csv(path_val, index=False)
        data_test.to_csv(path_test, index=False)

        drug_pairs = data[['drug_row', 'drug_col']].values
        edge_index = np.array([drug_pairs[:, 0], drug_pairs[:, 1]])
        np.save(edge_index_path, edge_index)

        i += 1


class load_data(Dataset):
    def __init__(self, csv_path, edge_index_path, transforms=None):
        data = pd.read_csv(csv_path)
        self.edge_index = np.load(edge_index_path)
        score_name = 'label'
        score_name_drop = 'synergy_loewe'
        self.labels = data[score_name]
        try:
            data = data.drop(columns=[score_name, score_name_drop, 'Unnamed: 0'])
        except KeyError:
            data = data.drop(columns=[score_name, score_name_drop])
        data = np.array(data)
        self.inputs = data
        self.transforms = transforms

    def __getitem__(self, index):
        labels = self.labels.iloc[index]
        inputs_np = self.inputs[index]
        inputs_tensor = torch.from_numpy(inputs_np).float()
        edge_index = torch.from_numpy(self.edge_index).long()
        return inputs_tensor, labels, edge_index

    def __len__(self):
        return len(self.labels)

