import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Koch et al.
        # Conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 64, 10)  # go from a single channel to 64 channels
        self.conv2 = nn.Conv2d(64, 128, 7)  
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    
    def convs(self, x):

        # Koch et al.
        # out_dim = in_dim - kernel_size + 1  
        #1, 105, 105
        x = F.relu(self.bn1(self.conv1(x)))  # starting with 1x105x105 image, x becomes 64x96x96
        x = F.max_pool2d(x, (2,2))  # 64x96x96 turns into 64x48x48
        x = F.relu(self.bn2(self.conv2(x)))  #64x48x48 turns into 128x42x42
        x = F.max_pool2d(x, (2,2))  # 128x42x42 turns into 128x21x21
        x = F.relu(self.bn3(self.conv3(x)))  # 128x21x21 turns into 128x18x18
        x = F.max_pool2d(x, (2,2))  # 128x18x18 turns into 128x9x9
        x = F.relu(self.bn4(self.conv4(x)))  # 128x9x9 turns into 256x6x6
        return x
    

    def forward(self, x1, x2):
        

        # Koch et al.
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256 * 6 * 6)
        x1 = self.sigmoid(self.fc1(x1))
        
        x2 = self.convs(x2)  # notice we use the same network
        x2 = x2.view(-1, 256 * 6 * 6)
        x2 = self.sigmoid(self.fc1(x2))

        # the outputs for img1 and img2 are "merged" by taking the absolute difference
        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x