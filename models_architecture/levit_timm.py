import torch
import timm
from torch import nn

class LeViT_128s(nn.Module):
    def __init__(self, num_classes=11, input_channels = 1 ,pretrained = True):
        super(LeViT_128s, self).__init__()
        self.base_model = timm.create_model('levit_128s', num_classes = 11, in_chans = 1 , pretrained=pretrained)
        # Modify the input layer

        # # print(self.base_model)
        # print(self.base_model.stem[0])
        # self.base_model.stem[0].linear = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        # print(self.base_model.head_dist)
        # # print(self.base_model.head_dist.linear.in_features)
        
        # # Modify the output layer

        # exit()

        # self.base_model.head_dist.linear = nn.Linear(self.base_model.head_dist.linear.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
        
class LeViT_128(nn.Module):
    def __init__(self, num_classes=11, input_channels = 1 ,pretrained = True):
        super(LeViT_128, self).__init__()
        self.base_model = timm.create_model('levit_128', num_classes = 11, in_chans = 1 , pretrained=pretrained)
        # Modify the input layer

        # # print(self.base_model)
        # print(self.base_model.stem[0])
        # self.base_model.stem[0].linear = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        # print(self.base_model.head_dist)
        # # print(self.base_model.head_dist.linear.in_features)
        
        # # Modify the output layer

        # exit()

        # self.base_model.head_dist.linear = nn.Linear(self.base_model.head_dist.linear.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


if __name__ == '__main__':
    model = LeViT_128s(pretrained = False)
    print(model)