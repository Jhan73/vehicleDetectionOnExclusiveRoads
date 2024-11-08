# pip install torch torchvision opencv-python

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, num_classes, input_channels=1, hidden_size=256):
        super(CRNN, self).__init__()
        
        # Capas convolucionales
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        
        # Capas LSTM bidireccionales
        self.rnn = nn.LSTM(256, hidden_size, bidirectional=True, batch_first=True)
        
        # Capa totalmente conectada
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Pasar por capas convolucionales
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Reshape para RNN
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = x.permute(0, 2, 1)  # Cambia a (batch, width, features)

        # Pasar por LSTM
        x, _ = self.rnn(x)
        
        # Pasar por capa totalmente conectada
        x = self.fc(x)

        return x





