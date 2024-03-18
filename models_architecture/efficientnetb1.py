import torch
import timm
from torch import nn

class EfficientNetB1(nn.Module):
    def __init__(self, num_classes=11, input_channels = 1 ,pretrained = True):
        super(EfficientNetB1, self).__init__()
        # Load pre-trained EfficientNetB1
        if(pretrained):
            print("Loading pre-trained EfficientNetB1")
        self.base_model = timm.create_model('efficientnet_b1', pretrained=pretrained, num_classes = num_classes)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.base_model(x)
        x = torch.nn.functional.log_softmax(x, dim=1)

        return x
        
class EfficientNetB1_custom(nn.Module):
    def __init__(self, num_classes=11, input_channels = 1 ,pretrained = True):
        super(EfficientNetB1_custom, self).__init__()
        # Load pre-trained EfficientNetB1
        self.base_model = timm.create_model('efficientnet_b1', pretrained=pretrained, num_classes = num_classes)

        # Modify the input layer
        original_conv_weights = self.base_model.conv_stem.weight
        new_conv_weights = original_conv_weights.mean(dim=1, keepdim=True)
        new_conv_weights = new_conv_weights.repeat(1, input_channels, 1, 1)
        self.base_model.conv_stem = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.base_model.conv_stem.weight = nn.Parameter(new_conv_weights)

        # Modify the output layer
        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        
        # x = x.repeat(1, 3, 1, 1)
        x = self.base_model(x)
        # x = torch.nn.functional.log_softmax(x, dim=1)

        return x