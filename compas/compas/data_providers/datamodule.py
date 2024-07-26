import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from compas.data_providers import TSSingleDataset, TSMultiDataset


# DataModule - Single Series
class TSSingleDataModule(pl.LightningDataModule):
    """
    DataModule - Single Series
        Dataset class for single-time-series dataset.
    ---
    (Parameters)
        data_path : (str) path to the data csv file
        emd_cd : (str) the EMD-code to use. A subset of the dataset with column EMD_CD == emd_cd will be used
        input_steps : (int) model input sequence length
        output_steps : (int) model output sequence length
        test_size : (int) size of test set
        val_size : (int) size of validation set
        batch_size : (int) batch size
        x_cols : (List[str]) list of column names to use from the dataframe(s)
    """

    def __init__(
        self,
        data_path: str,
        emd_cd: str,
        input_steps: int,
        output_steps: int,
        test_size: int = 12,
        val_size: int = 12,
        batch_size: int = 8,
        x_cols: list = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.emd_cd = emd_cd
        self.x_cols = x_cols
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        df = pd.read_csv(self.data_path, low_memory=False)
        df.dropna(how="any", inplace=True)
        self.dataframe = df.loc[df["EMD_CD"] == self.emd_cd].sort_values(by="STD_YM")
        if not self.x_cols is not None:
            self.x_cols = list(df.columns[2:])  # exclude index columns

        # Y (vacancy_rate) column must be at the front
        if "vacancy_rate" not in self.x_cols:
            self.x_cols = ["vacancy_rate"] + self.x_cols
        if self.x_cols[0] != "vacancy_rate":
            self.x_cols.insert(0, self.x_cols.pop(self.x_cols.index("vacancy_rate")))

    def setup(self, stage: str = None) -> None:
        # (train, val, test) -> (ANY, 12, 12)
        train, test = train_test_split(
            self.dataframe,
            test_size=self.test_size,
            shuffle=False,
        )

        if self.val_size:
            train, val = train_test_split(train, test_size=self.val_size, shuffle=False)

        # scaler is set from train-set only
        self.train = TSSingleDataset(
            train, self.x_cols, self.input_steps, self.output_steps
        )
        self.scaler = self.train.scaler
        self.n_features = self.train.n_features
        self.test = TSSingleDataset(
            test, self.x_cols, self.input_steps, self.output_steps, scaler=self.scaler
        )

        if self.val_size:
            self.validation = TSSingleDataset(
                val,
                self.x_cols,
                self.input_steps,
                self.output_steps,
                scaler=self.scaler,
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)


# DataModule - Multi Series
class TSMultiDataModule(pl.LightningDataModule):
    """
    DataModule - Single Series
        Dataset class for single-time-series dataset.
    ---
    (Parameters)
        data_path : (str) path to the data csv file
        input_steps : (int) model input sequence length
        output_steps : (int) model output sequence length
        test_size : (int) size of test set
        val_size : (int) size of validation set
        batch_size : (int) batch size
        x_cols : (List[str]) list of column names to use from the dataframe(s)
    """

    def __init__(
        self,
        data_path: str,
        input_steps: int,
        output_steps: int,
        test_size: int = 12,
        val_size: int = 12,
        batch_size: int = 8,
        x_cols: list = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.x_cols = x_cols
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        df = pd.read_csv(self.data_path, low_memory=False)
        df.dropna(how="any", inplace=True)
        df.set_index("EMD_CD", inplace=True)
        self.df_list = [
            df.loc[emd].reset_index(drop=False).sort_values(by="STD_YM")
            for emd in df.index.unique()
        ]
        if not self.x_cols:
            self.x_cols = list(self.df_list[0].columns[2:])  # exclude index columns

        # Y (vacancy_rate) column must be at the front
        if "vacancy_rate" not in self.x_cols:
            self.x_cols = ["vacancy_rate"] + self.x_cols
        if self.x_cols[0] != "vacancy_rate":
            self.x_cols.insert(0, self.x_cols.pop(self.x_cols.index("vacancy_rate")))

    def setup(self, stage: str = None) -> None:
        # (train, val, test) -> (ANY, 12, 12)
        splits = [
            train_test_split(df, test_size=self.test_size, shuffle=False)
            for df in self.df_list
        ]
        trains_t, tests = [x[0] for x in splits], [x[1] for x in splits]

        if self.val_size:
            splits = [
                train_test_split(df, test_size=self.val_size, shuffle=False)
                for df in trains_t
            ]
            trains, vals = [x[0] for x in splits], [x[1] for x in splits]
        else:
            trains = trains_t

        # scaler is set from train-set only
        self.train = TSMultiDataset(
            trains,
            x_cols=self.x_cols,
            input_steps=self.input_steps,
            output_steps=self.output_steps,
        )
        self.scaler = self.train.scaler
        self.n_features = self.train.n_features
        self.test = TSMultiDataset(
            tests,
            x_cols=self.x_cols,
            input_steps=self.input_steps,
            output_steps=self.output_steps,
            scaler=self.scaler,
        )

        if self.val_size:
            self.validation = TSMultiDataset(
                vals,
                x_cols=self.x_cols,
                input_steps=self.input_steps,
                output_steps=self.output_steps,
                scaler=self.scaler,
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
