import torch
import torch.nn as nn

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
    def __init__(self, input_size, output_size, hidden_size, kernel_size, dropout):
        super(CNN1DSimple, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        conv_output_size = input_size - kernel_size + 1
        self.fc = nn.Linear(hidden_size * conv_output_size, output_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x

