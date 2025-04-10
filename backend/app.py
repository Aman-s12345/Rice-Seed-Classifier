import io
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from flask_cors import CORS

# Load trained model (same architecture as before)
class BasicCNN(torch.nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        f1 = x
        x = self.pool(torch.relu(self.conv2(x)))
        f2 = x
        x = self.pool(torch.relu(self.conv3(x)))
        f3 = x
        return f1, f2, f3

class FPN(torch.nn.Module):
    def __init__(self, num_classes):
        super(FPN, self).__init__()
        self.backbone = BasicCNN()
        self.lat_conv1 = torch.nn.Conv2d(64, 256, kernel_size=1)
        self.lat_conv2 = torch.nn.Conv2d(128, 256, kernel_size=1)
        self.lat_conv3 = torch.nn.Conv2d(256, 256, kernel_size=1)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = torch.nn.Conv2d(256, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)
        p3 = self.lat_conv3(f3)
        p2 = self.upsample(p3) + self.lat_conv2(f2)
        p1 = self.upsample(p2) + self.lat_conv1(f1)
        out = self.final_conv(p1)
        return out.mean([2, 3])

# Load model
num_classes = 10  # Change this based on your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FPN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("fpn_model.pth", map_location=device))
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Flask app
app = Flask(__name__)
CORS(app)

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

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)  # Preprocess
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # Get prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()
        class_name = CLASS_LABELS[class_index]  # Map index to label

    return jsonify({
        "predicted_class_index": class_index,
        "predicted_class_name": class_name
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
