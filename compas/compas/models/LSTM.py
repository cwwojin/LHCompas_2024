import torch
import torch.nn as nn


# Simple LSTM model
class LSTMSimple(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        input_steps,
        output_steps,
        dropout=0,
        bidirectional=False,
        scaler=None,
    ):
        super(LSTMSimple, self).__init__()
        self.input_size = input_size
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.output_size = output_steps * input_size  # input_size == N_FEATURES
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.scaler = scaler
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(
            hidden_size * (2 if self.bidirectional else 1), self.output_size
        )

    def forward(self, x):
        # hidden state
        h_0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_size,
        ).to(x.device)
        c_0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_size,
        ).to(x.device)

        # forward lstm & fcn
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
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
                y_hat = self(current_input)  # y_hat.shape == (B, L_out * N)
                pred = y_hat.reshape(-1, self.output_steps, self.input_size)
                forecast_series.append(pred)
                current_input = torch.cat(
                    (current_input[:, self.output_steps :, :], pred), dim=1
                )

            # Combine the forecasts into a continuous time series
            out = torch.cat(forecast_series, dim=1).squeeze(0)

        return out
