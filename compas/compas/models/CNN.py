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
        output : [B, L_out, C]
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
        x = x.permute(0, 2, 1)  # (B, in_channels, input_steps)
        out = self.layers(x)  # out : (B, output_steps * in_channels)
        out = out.reshape(-1, self.in_channels, self.output_steps)
        out = out.permute(0, 2, 1)
        return out

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
                pred = self(current_input)  # pred.shape == (B, L_out, N)
                forecast_series.append(pred)
                current_input = torch.cat(
                    (current_input[:, self.output_steps :, :], pred), dim=1
                )

            # Combine the forecasts into a continuous time series
            out = torch.cat(forecast_series, dim=1).squeeze(0)

        return out[:steps, :]
