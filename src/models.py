import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class DeepConvClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, p_drop=0.3), # ドロップアウト率の調整
            ConvBlock(hid_dim, hid_dim * 2, p_drop=0.3),
            ConvBlock(hid_dim * 2, hid_dim * 4, p_drop=0.3),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 4, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same") # 畳み込み層1
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same") # 畳み込み層2
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim) # バッチ正規化層1
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim) # バッチ正規化層2
        
        self.dropout = nn.Dropout(p_drop) # ドロップアウト層

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # 入力と出力が同じ次元のときスキップ
        else:
            X = self.conv0(X) # 通常
        
        X = F.gelu(self.batchnorm0(X)) # 畳み込み層1 -> バッチ正規化 -> GELU活性化関数
        X = self.conv1(X) + X  # 畳み込み層2 > スキップ
        X = F.gelu(self.batchnorm1(X)) # バッチ正規化 -> GELU活性化関数
        
        return self.dropout(X) # ドロップアウト層