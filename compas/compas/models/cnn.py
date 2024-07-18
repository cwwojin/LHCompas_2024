import torch
import torch.nn as nn
from typing import List

"""
    Convolution Neural Network
    ---
    (Parameters)
        input_size : input sequence length (num of time steps)
        output_size : output sequence length
        hidden_size : num of neurons in fully connected layer
        kernel_size : kernel size for CNN
            kernel_size < input_steps
        dropout : dropout probability

    ---
    (Dimensions)
        input : [B, L_in, C]
            B - Batch size, L_in - input length, C - channels
        output : [B, L_out, C]
            B - Batch size, L_out - output length, C - channels
    """


class CNN1DSimple(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        kernel_size,
        dropout,
        scaler=None,
        feature_names: List[str] = [],
    ):
        super(CNN1DSimple, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=hidden_size, kernel_size=kernel_size, stride=1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        conv_output_size = input_size - kernel_size + 1
        self.fc = nn.Linear(hidden_size * conv_output_size, output_size)
        self.feature_names = feature_names
        self.scaler = scaler

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

    @torch.no_grad()
    def predict(self, x, steps: int):
        """
        Perform long-term forecast on the input series.
        This method is done on unbatched input only.

        Parameters:
            x : input tensor of shape (L_in, N)
                L_in - input sequence length
                N - feature dimension
            steps : integer indicating the number of time steps to predict

        Returns:
            out : output tensor of shape (steps, N)
        """
        # Initialize the list to store the forecasted series
        forecast_series = []
        current_input = x.unsqueeze(0)  # Add batch dimension

        for _ in range(steps):
            y_hat = self(current_input)  # y_hat.shape == (B, L_out)
            forecast_series.append(y_hat.squeeze(0))  # Remove batch dimension
            current_input = torch.cat(
                (current_input[:, :, 1:], y_hat.unsqueeze(2)), dim=2
            )

        # Combine the forecasts into a continuous time series
        out = torch.cat(forecast_series, dim=0)

        return out
