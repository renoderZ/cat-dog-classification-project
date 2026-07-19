from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
# Create upload folder if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Restrict allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def build_model():
    # Match your training model architecture perfectly
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Freeze backbone layers same as training
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    # Same fully connected head
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model

# Initialize model and load weights safely
model = build_model()
# Add weights file existence check to avoid crash
weight_path = 'cat_dog_resnet18.pth'
if os.path.exists(weight_path):
    # Force CPU load, avoid CUDA missing error
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
else:
    raise FileNotFoundError(f"Model weight file {weight_path} not found! Put it in the same folder as this script.")

# Disable gradient calculation for inference
model.eval()

# Image preprocessing transform (matches training normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file part exists in form data
        if 'file' not in request.files:
            return "No file selected, go back and upload an image."
        
        file = request.files['file']
        # Empty filename check
        if file.filename == '':
            return "File name is empty, please select a valid image."
        
        # Only process supported image formats
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            try:
                # Open image and convert to RGB to handle transparent PNG
                img = Image.open(filepath).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                
                # Inference without computing gradients (save memory)
                with torch.no_grad():
                    output = model(img_tensor)
                    pred_index = torch.argmax(output, dim=1).item()
                    pred_label = "Cat" if pred_index == 0 else "Dog"
                
                return f"<h2>Prediction Result: {pred_label}</h2>"
            except Exception as e:
                return f"Image processing failed: {str(e)}"
        else:
            return "Unsupported file type! Only png, jpg, jpeg, bmp are allowed."
    # GET request: render upload page
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)