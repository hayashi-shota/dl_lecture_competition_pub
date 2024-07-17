import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable
from termcolor import cprint
import scipy.signal

# 前処理関数の定義
def preprocess_eeg(eeg_data):
    # リサンプリング（250Hz）
    eeg_data = scipy.signal.resample(eeg_data, 250, axis=1)
    
    # フィルタリング（1-50Hz）
    sos = scipy.signal.butter(10, [1, 50], btype='bandpass', output='sos', fs=250)
    eeg_data = scipy.signal.sosfilt(sos, eeg_data, axis=1)
    
    # スケーリング（標準化）
    eeg_data = (eeg_data - eeg_data.mean(axis=1, keepdims=True)) / eeg_data.std(axis=1, keepdims=True)
    
    # ベースライン補正（開始時の平均を引く）
    baseline = eeg_data[:, :50].mean(axis=1, keepdims=True)
    eeg_data = eeg_data - baseline
    
    return eeg_data

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform: Optional[Callable] = None, apply_preprocessing: bool = True) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        self.transform = transform
        self.apply_preprocessing = apply_preprocessing

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            sample = (self.X[i], self.y[i], self.subject_idxs[i])
        else:
            sample = (self.X[i], self.subject_idxs[i])
        
        # 前処理を適用（元のデータを変更しないようにコピーを使用）
        if self.apply_preprocessing:
            processed_data = preprocess_eeg(sample[0].numpy())
            sample = (torch.tensor(processed_data, dtype=torch.float), *sample[1:])
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
