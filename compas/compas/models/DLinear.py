"""
This model implementation is sourced from :
    https://github.com/cure-lab/LTSF-Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    ---
    (Parameters)
        input_steps : input sequence length
        output_steps : output sequence length
        kernel_size : kernel size for series decomposition block
            kernel_size < input_steps
        individual : (bool)
        channels : channels

    ---
    (Dimensions)
        input : [B, L_in, C]
            B - Batch size, L_in - input length, C - channels
        output : [B, L_out, C]
            B - Batch size, L_out - output length, C - channels

    """

    def __init__(
        self,
        input_steps: int,
        output_steps: int,
        channels: int,
        kernel_size: int = 25,
        individual: bool = False,
        scaler: Any = None,
    ):
        super(DLinear, self).__init__()
        self.input_steps = input_steps
        self.output_steps = output_steps

        # Decompsition Kernel Size
        self.kernel_size = kernel_size
        self.decompsition = series_decomp(self.kernel_size)
        self.individual = individual
        self.channels = channels

        # Scaler
        self.scaler = scaler

        # Seasonal & Trend Components
        self.Linear_Seasonal = nn.Linear(self.input_steps, self.output_steps)
        self.Linear_Trend = nn.Linear(self.input_steps, self.output_steps)

        self.Linear_Seasonal_i = nn.ModuleList()
        self.Linear_Trend_i = nn.ModuleList()
        if self.individual:
            for _ in range(self.channels):
                self.Linear_Seasonal_i.append(
                    nn.Linear(self.input_steps, self.output_steps)
                )
                self.Linear_Trend_i.append(
                    nn.Linear(self.input_steps, self.output_steps)
                )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.output_steps],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.output_steps],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i, seasonal in enumerate(self.Linear_Seasonal_i):
                seasonal_output[:, i, :] = seasonal(seasonal_init[:, i, :])
            for i, trend in enumerate(self.Linear_Trend_i):
                trend_output[:, i, :] = trend(trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

    @torch.jit.export
    def predict(self, x, steps: int):
        """
        Perform long-term forecast on the input series.
        This method is done on unbatched input only

        ---
        (Parameters)
            x : input tensor of shape (L_in, N)
                L_in - input sequence length
                N - feature dimension
            steps : integer indicating the number of time steps to predict
                (default) None, which will use the model default output_steps value

        ---
        (Returns)
            out : output tensor of shape (L_out, N)
        """
        with torch.no_grad():
            # Initialize the list to store the forecasted series
            forecast_series = []
            forecast_steps = (steps // self.output_steps) + (
                1 if (steps % self.output_steps) else 0
            )

            # Iteratively forecast the next time steps
            current_input = x[-self.input_steps :, :].unsqueeze(0)
            for _ in range(forecast_steps):
                pred = self(current_input)  # y_hat.shape == (B, L_out, C)
                forecast_series.append(pred)
                current_input = torch.cat(
                    (current_input[:, self.output_steps :, :], pred), dim=1
                )

            # Combine the forecasts into a continuous time series
            out = torch.cat(forecast_series, dim=1).squeeze(0)

        return out[:steps, :]
