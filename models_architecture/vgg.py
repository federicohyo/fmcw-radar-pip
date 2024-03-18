import torch
import torch.nn as nn

class FP32Traceble(nn.Module):
    def forward(self, x):
        x = traceable_code(x)

class SimpleVGG(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000):
        super(SimpleVGG, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # nn.ReLU(),
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # nn.ReLU(),
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2),

        # Third Convolutional Block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # nn.ReLU(),
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # nn.ReLU(),
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2)

        # self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(256 * 8 * 8, 4096)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)


    def forward(self, x):
        # print(x.shape)
        # x = self.features(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.maxpool2d(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.maxpool2d(x)


        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)

        return x
