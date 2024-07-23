import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Union, List


# Dataset - Single Series
class TSSingleDataset(Dataset):
    """
    Dataset - Single Series
        Dataset class for single time-series dataset.
    ---
    (Parameters)
        data : (DataFrame) initial dataframe
        x_cols : (List[str]) list of column names to use from the dataframe
        input_steps : (int) model input sequence length
        output_steps : (int) model output sequence length
        scaler : (StandardScaler) scaler to use for dataset preprocessing (default = None)
            setting this parameter will use the pre-trained scaler without fitting
    """

    def __init__(self, data, x_cols, input_steps, output_steps, scaler=None):
        self.dataframe = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.input_steps = input_steps
        self.output_steps = output_steps

        # setup scaler
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(self.dataframe[x_cols])

        # prepare data
        self.data = torch.tensor(
            self.scaler.transform(self.dataframe[x_cols]),
            dtype=torch.float32,
        )
        self.n_features = self.data.shape[1]  # feature-dim

    def __len__(self):
        return len(self.data) - self.input_steps - self.output_steps + 1

    def __getitem__(self, idx):
        # x : (input_steps, n_features)
        x = self.data[idx : idx + self.input_steps, :]
        y = self.data[
            idx + self.input_steps : idx + self.input_steps + self.output_steps, :
        ]
        return x, y


# Dataset - Multi Series
class TSMultiDataset(Dataset):
    """
    Dataset - Multi Series
        Dataset class for multiple-time-series dataset.
    ---
    (Parameters)
        data : (List[DataFrame]) list of initial dataframe(s).
            each dataframe is interpreted as individual time-series datasets.
            The combined dataset will be generated from these dataframes.
        x_cols : (List[str]) list of column names to use from the dataframe(s)
        input_steps : (int) model input sequence length
        output_steps : (int) model output sequence length
        scaler : (StandardScaler) scaler to use for dataset preprocessing (default = None)
            setting this parameter will use the pre-trained scaler without fitting
    """

    def __init__(
        self,
        data: Union[List[pd.DataFrame], pd.DataFrame],
        x_cols,
        input_steps,
        output_steps,
        scaler=None,
    ):

        # data : list of dataframe or single dataframe (all same shape / time series length)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.df_list = [data] if isinstance(data, pd.DataFrame) else data
        self.df_combined = pd.concat(self.df_list, axis=0)
        self.series_length = (
            self.df_list[0].shape[0] - self.input_steps - self.output_steps + 1
        )  # length of each series

        # setup scaler
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df_combined[x_cols])

        # dim : (N_series, N_timesteps, N_features)
        self.data = torch.tensor(
            np.asarray([self.scaler.transform(df[x_cols]) for df in self.df_list]),
            dtype=torch.float32,
        )
        self.n_features = self.data.shape[2]
        self.shuffled = False

    def __len__(self):
        return self.series_length * len(self.df_list)

    def __getitem__(self, idx):
        idx_1 = idx // self.series_length
        idx_2 = idx % self.series_length

        # shape : X - (input_steps, N_features), Y - (output_steps, N_features)
        x = self.data[idx_1, idx_2 : idx_2 + self.input_steps, :]
        y = self.data[
            idx_1,
            idx_2 + self.input_steps : idx_2 + self.input_steps + self.output_steps,
            :,
        ]
        return x, y

    def permute_feature(self, k: int = 0):
        # permute one column / feature of the dataset (the k-th column)
        if not self.shuffled:
            self.data_original = self.data.numpy()
        permute = self.data.numpy()
        np.random.shuffle(permute[:, :, k])
        self.data = torch.tensor(permute, dtype=torch.float32)
        self.shuffled = True

    def revert_permutation(self):
        if self.shuffled:
            self.data = torch.tensor(self.data_original, dtype=torch.float32)
        self.shuffled = False
