import torch
import torch.nn as nn

class CNN1DSimple(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, kernel_size=3, dropout=0.1):
        super(CNN1DSimple, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x
