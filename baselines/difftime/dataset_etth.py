import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

# 35 attributes which contains enough non-values


def extract_hour(x):
    h, _ = map(int, x.split(":"))
    return h



class etth_Dataset(Dataset):
    def __init__(self, eval_length=24, dataname=None, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        data_name = dataname
        if dataname=='etth':
            path = ("./data/etth/ETTh.csv")
            df = pd.read_csv(path, header=0)
            df.drop(df.columns[0], axis=1, inplace=True)
            data = df.values            
        # elif dataname=='etth_mp':
        #     path = ("./data/morning_peak_etth.csv")
        # elif dataname=='etth_ep':
        #     path = ("./data/evening_peak_etth.csv")            
        # elif dataname=='energy_mp':
        #     path = ("./data/morning_peak_energy.csv")
        # elif dataname=='energy_ep':
        #     path = ("./data/evening_peak_energy.csv")            
                 


        
        elif data_name == 'etth_mp':
            data = pd.read_csv(('data/morning_peak_etth.csv'), header=0)
            data.drop(data.columns[0], axis=1, inplace=True)
            data = data.values
        elif data_name == 'etth_ep':
            data = pd.read_csv(('data/evening_peak_etth.csv'), header=0)
            data.drop(data.columns[0], axis=1, inplace=True)
            data = data.values    
        elif data_name == 'etth':
            data = pd.read_csv(('data/ETTh.csv'), header=0)
            data.drop(data.columns[0], axis=1, inplace=True)
            data = data.values           
        elif data_name == 'energy_mp':
            data = pd.read_csv(('data/morning_peak_energy.csv'), header=0)
            # ori_data.drop(ori_data.columns[0], axis=1, inplace=True)
            data = data.values        
        elif data_name == 'energy_ep':
            data = pd.read_csv(('data/evening_peak_energy.csv'), header=0)
            # ori_data.drop(ori_data.columns[0], axis=1, inplace=True)
            data = data.values    
        elif data_name == 'pems04_mp':
            data = pd.read_csv(('data/morning_peak_pems04.csv'), header=0)
            # ori_data.drop(ori_data.columns[0], axis=1, inplace=True)
            data = data.values        
        elif data_name == 'pems04_ep':
            data = pd.read_csv(('data/evening_peak_pems04.csv'), header=0)
            # ori_data.drop(ori_data.columns[0], axis=1, inplace=True)
            data = data.values  
        elif data_name == 'pems08_mp':
            data = pd.read_csv(('data/morning_peak_pems08.csv'), header=0)
            # ori_data.drop(ori_data.columns[0], axis=1, inplace=True)
            data = data.values        
        elif data_name == 'pems08_ep':
            data = pd.read_csv(('data/evening_peak_pems08.csv'), header=0)
            # ori_data.drop(ori_data.columns[0], axis=1, inplace=True)
            data = data.values                                
        elif data_name == 'energy':
            data = np.loadtxt(('data/energy_data.csv'), delimiter=",", skiprows=1)

        print('data name is', dataname)       

        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        data = scaler.transform(data)
        print('data shape is ',data.shape)
        data_single = []
        for i in range(data.shape[0]-eval_length+1):
            data_single.append(data[i:i+eval_length])
            
        self.observed_values = np.array(data_single)
        b,t,n = self.observed_values.shape
        self.observed_values = self.observed_values.transpose(2,0,1).reshape(b * n, t, 1)
        # ori_data.transpose(2, 0, 1).reshape(b * n, t, 1)
        print('observed_values shape is ',self.observed_values.shape)

        save_dir = os.path.join('./save',dataname)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"ori_{dataname}{eval_length}.npy")
        np.save(save_path, self.observed_values)
        print(f"{dataname} dataset is saved.")

        self.observed_masks = ~np.isnan(self.observed_values)
        print('observed_masks shape is ',self.observed_masks.shape)

        masks = self.observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices] = False
        self.gt_masks = masks.reshape(self.observed_masks.shape)
        print('gt_masks shape is ',self.gt_masks.shape)
        self.observed_values = np.nan_to_num(self.observed_values)
        self.observed_masks = self.observed_masks.astype("float32")
        self.gt_masks = self.gt_masks.astype("float32")
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(eval_length=24, dataname=None, seed=1, nfold=None, batch_size=32, missing_ratio=0.0):

    # only to obtain total length of dataset
    dataset = etth_Dataset(eval_length=eval_length,dataname=dataname,missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    # test_index = indlist[start:end]

    test_index = indlist

    # remain_index = np.delete(indlist, np.arange(start, end))
    remain_index = indlist

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    print('len(dataset)',len(dataset))
    num_train = (int)(len(dataset) * 0.8)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = etth_Dataset(
        eval_length=eval_length,dataname=dataname,use_index_list=train_index, missing_ratio=0.0, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = etth_Dataset(
        eval_length=eval_length,dataname=dataname,use_index_list=valid_index, missing_ratio=1, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = etth_Dataset(
        eval_length=eval_length,dataname=dataname,use_index_list=test_index, missing_ratio=1, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader
