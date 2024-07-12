import torch
import torch.nn as nn


# Simple LSTM model
class LSTMSimple(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_steps,
        bidirectional=False,
        scaler=None,
    ):
        super(LSTMSimple, self).__init__()
        self.input_size = input_size
        self.output_steps = output_steps
        self.output_size = output_steps * input_size  # input_size == N_FEATURES
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.scaler = scaler
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=0,
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
