import os
import os.path as path
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dateutil.relativedelta import relativedelta


class ForecastModel:
    """
    Time-Series Forecast Model Class for Inference
    ---
    (Parameters)
        model_path : (str) directory name of the model. Directory should contain 2 files `model.pt`, `scaler.pkl` to load

    """

    def __init__(self, model_path: str):
        # load model & scaler from `model_path`
        self.model = torch.jit.load(path.join(model_path, "model.pt"))
        with open(path.join(model_path, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        # feature names
        self.feature_names = self.model.feature_names

    def get_input_from_df(self, df: pd.DataFrame):
        """
        Preprocess input DataFrame into model input tensor(s)
        ---
        (Parameters)

        ---
        (Returns)

        """
        out = torch.tensor(
            self.scaler.transform(df[self.feature_names]),
            dtype=torch.float32,
        )
        return out

    def predict(self, x, steps: int):
        out = self.model.predict(x, steps)
        return out

    def forecast(self, x, steps: int, vacancy: bool = False):
        """
        Perform long-term forecast on the input time series data.
        ---
        (Parameters)
            x : [DataFrame, np.array] input time series data of shape (L_in, M)
                L_in - input sequence length
                M - number of columns. should be larger than `N=model.n_features` & contain all of `self.feature_names`
                    the DataFrame should include columns ['EMD_CD', 'STD_YM']
                    If x is given as np.array, then M == N.
            steps : integer indicating the number of time steps to predict

        ---
        (Returns)
            out : (DataFrame) output data of shape (L_out, N)

        """
        if isinstance(x, pd.DataFrame):
            x = x.sort_values(by="STD_YM")
            model_in = self.get_input_from_df(x)
        else:
            model_in = torch.tensor(self.scaler.transform(x), dtype=torch.float32)

        model_out = self.predict(model_in, steps)  # shape : (steps, N_features)

        # reverse-transform w/ scaler
        out_scaled = self.scaler.inverse_transform(model_out.numpy())
        df_out = pd.DataFrame(out_scaled, columns=self.feature_names)

        # set date index if given
        if isinstance(x, pd.DataFrame):
            max_date = datetime.strptime(x["STD_YM"].max(), "%Y-%m")
            forecast_dates = pd.date_range(
                start=max_date + relativedelta(months=1),
                end=max_date + relativedelta(months=steps),
                freq="MS",
            ).strftime("%Y-%m")
            df_out = df_out.set_index(forecast_dates)

        return df_out["vacancy_rate"] if vacancy else df_out
