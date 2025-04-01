# Image Classifier Web Application

A web application that allows users to upload images and get classification results using a PyTorch Feature Pyramid Network (FPN) model.

## Project Structure

```
.
├── backend/                  # Flask backend
│   ├── app.py                # Main Flask application
│   ├── create_sample_model.py # Script to create a sample model
│   └── models/               # Directory to store the PyTorch model
│       └── fpn_model.pth     # The trained FPN model file
└── frontend/                 # React frontend
    ├── public/               # Public assets
    │   └── index.html        # HTML template
    └── src/                  # React source code
        ├── App.js            # Main application component
        ├── App.css           # Styles for the application
        ├── index.js          # Entry point
        └── components/       # React components
            └── ImageUpload.js # Component for image upload and display
```

## Prerequisites

- Python 3.6+
- Node.js and npm
- PyTorch
- Flask
- React

## Setup and Running

### Backend Setup

1. Install Python dependencies:
   ```
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install flask flask-cors torch torchvision Pillow numpy
   ```

2. Place your trained `fpn_model.pth` in the backend/models directory, or generate a sample model (for demonstration purposes):
   ```
   python create_sample_model.py
   ```

3. Run the Flask server:
   ```
   python app.py
   ```
   The server will start at http://localhost:5000

### Frontend Setup

1. Install JavaScript dependencies:
   ```
   cd frontend
   npm install
   ```

2. Start the React development server:
   ```
   npm start
   ```
   The application will be available at http://localhost:3000

## Usage

1. Open the web application at http://localhost:3000
2. Click on the "Choose File" button to select an image
3. Click "Classify Image" to send the image to the backend for classification
4. View the classification results displayed on the page

## Customizing the Model

If you have a different FPN model:

1. Update the model class labels in `app.py` to match your model's output classes
2. If your model has a different architecture, update the `FPNModel` class in `app.py` accordingly

## Notes

- The application uses a Feature Pyramid Network (FPN) for image classification
- The FPN architecture enhances feature extraction by combining features at different scales
- For production use, consider optimizing the model and using a production-ready server 