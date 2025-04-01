import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to your trained model
MODEL_PATH = os.path.join('models', 'fpn_model.pth')

# Define CNN Backbone
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        f1 = x
        x = self.pool(torch.relu(self.conv2(x)))
        f2 = x
        x = self.pool(torch.relu(self.conv3(x)))
        f3 = x
        return f1, f2, f3

# Define Feature Pyramid Network (FPN) Head
class FPN(nn.Module):
    def __init__(self, num_classes=10):  # Changed back to 10 classes to match trained model
        super(FPN, self).__init__()
        self.backbone = BasicCNN()
        self.lat_conv1 = nn.Conv2d(64, 256, kernel_size=1)
        self.lat_conv2 = nn.Conv2d(128, 256, kernel_size=1)
        self.lat_conv3 = nn.Conv2d(256, 256, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)
        p3 = self.lat_conv3(f3)
        p2 = self.upsample(p3) + self.lat_conv2(f2)
        p1 = self.upsample(p2) + self.lat_conv1(f1)
        out = self.final_conv(p1)
        return out.mean([2, 3])

# Define the class labels globally - use only the first 10 since model has 10 classes
CLASS_LABELS = [
  "Blwrc -GA",
  "Bwr-2",
  "Jyothi",
  "Kau Manu ratna(km-1)",
  "Menu verna",
  "Pour nami (p-1)",
  "Sreyas",
  "Uma-1"
]

# Create the models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load model
try:
    model = FPN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("FPN model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # You might want to terminate the application here if model loading fails
    # Or use a fallback model

# Image preprocessing - resize to fixed dimensions to avoid size mismatches
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fixed dimensions
        transforms.ToTensor(),
    ])
    
    image = Image.open(io.BytesIO(image_bytes))
    
    
    return transform(image).unsqueeze(0)  # Add batch dimension

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    
    if not image_bytes:
        return jsonify({'error': 'Empty image'}), 400
    
    try:
        tensor = preprocess_image(image_bytes)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        print(predicted_class.item())
        return jsonify({
            'class': CLASS_LABELS[predicted_class.item()],
            'confidence': confidence.item()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 