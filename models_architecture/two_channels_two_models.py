import torch
from torch import nn

class TwoChannelsTwoModels(nn.Module):
    def __init__(self, basemodel, num_classes, combined_features = 100):
        super(TwoChannelsTwoModels, self).__init__()
        self.channel1_model = basemodel
        self.channel2_model = basemodel
        
        # Define the small neural network for combining outputs
        self.combine = nn.Linear(2 * num_classes, combined_features)
        self.final_layer = nn.Linear(combined_features, num_classes)
    
    def forward(self, x):
        # Split the input into two channels
        x1 = x[:,0:1, :, :]
        x2 = x[:,1:2, :, :]

        # Process each channel
        out1 = self.channel1_model(x1)
        out2 = self.channel2_model(x2)

        # Combine outputs
        combined = torch.cat((out1, out2), dim=1)

        # Pass through the final neural network
        combined = self.combine(combined)
        combined = self.final_layer(combined)
        return combined