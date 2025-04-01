import os
import torch
from torch import nn

# Same FPN model architecture as defined in app.py
class FPNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(FPNModel, self).__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FPN lateral connections
        self.lateral1 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(64, 256, kernel_size=1)
        
        # FPN top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Bottom-up pathway
        c1 = self.conv1(x)
        c1 = self.relu1(c1)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        c2 = self.relu2(c2)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        c3 = self.relu3(c3)
        p3 = self.pool3(c3)
        
        # Top-down pathway and lateral connections
        p3_out = self.lateral1(p3)
        
        p2_lat = self.lateral2(c2)
        p2_out = p2_lat + self.upsample(p3_out)
        
        p1_lat = self.lateral3(c1)
        p1_out = p1_lat + self.upsample(p2_out)
        
        # Classification
        out = self.avgpool(p3_out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

# Create and save a sample FPN model
def create_sample_model():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Initialize model
    model = FPNModel()
    
    # Save the model
    torch.save(model.state_dict(), os.path.join('models', 'fpn_model.pth'))
    print("Sample FPN model created and saved to models/fpn_model.pth")

if __name__ == "__main__":
    create_sample_model() 