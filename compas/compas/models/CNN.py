import torch
import torch.nn as nn
from typing import List


class CNN1DSimple(nn.Module):
    """
    Convolution Neural Network
    ---
    (Parameters)
        in_channels : number of features in input / output (n_features)
        input_steps : input sequence length
        output_steps : output sequence length (number of time steps to predict)
        hidden_size : number of output channels of 1D-convolution
        kernel_size : kernel size of 1D-convolution
        dropout : dropout rate
        activation : type of activation. one of - 'relu', 'tanh'
        scaler : dataset scaler
        feature_names : list of data feature names

    ---
    (Dimensions)
        input : [B, L_in, C]
            B - Batch size, L_in - input length, C - n_features
        output : [B, L_out * C]
            B - Batch size, L_out - output length, C - n_features

        (conv1D layer output) : [B, (input_steps - kernel_size + 1), hidden_size]

    """

    def __init__(
        self,
        in_channels,
        input_steps,
        output_steps,
        hidden_size,
        kernel_size,
        dropout,
        activation="relu",
        scaler=None,
        feature_names: List[str] = [],
    ):
        super(CNN1DSimple, self).__init__()
        assert activation in {
            "relu",
            "tanh",
        }, "activation should be one of : 'relu', 'tanh'"
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.output_size = output_steps * in_channels  # in_channels == n_features
        self.activation_type = activation
        self.dropout = dropout
        self.feature_names = feature_names

        # conv_output_size = input_steps - kernel_size + 1
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=kernel_size,
            ),
            nn.Tanh() if self.activation_type == "tanh" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_size * (input_steps - kernel_size + 1), self.output_size),
        )
        self.scaler = scaler

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.layers(x)
        return out

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
