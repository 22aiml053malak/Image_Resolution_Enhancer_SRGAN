from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import torch
import numpy as np
import RRDBNet_arch as arch  # Ensure this module is available

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
RESULT_FOLDER = 'results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Define the load_model function
def load_model(model_name, model_dir, device):
    model_path = os.path.join(model_dir, model_name)
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)  # Define the architecture
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)
    return model

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'RRDB_ESRGAN_x4.pth'  # Use the ESRGAN model
model = load_model(model_name, r'C:/Users/malak/Downloads/models', device)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Super-resolution process
            result_img_path = super_resolution(filepath, device, model)
            low_res_img_path = filepath
            
            # Pass paths to the result page
            return render_template('result.html', 
                                   low_res_img=os.path.basename(low_res_img_path), 
                                   high_res_img=os.path.basename(result_img_path))

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def send_low_res_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def send_high_res_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

def super_resolution(path_img, device, model):
    base = os.path.splitext(os.path.basename(path_img))[0]
    img = cv2.imread(path_img)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    result = np.transpose(result[[2, 1, 0], :, :], (1, 2, 0))
    result = (result * 255.0).round()
    result_path = os.path.join(RESULT_FOLDER, f'{base}_sr.png')
    cv2.imwrite(result_path, result)
    return result_path

if __name__ == "__main__":
    app.run(debug=True)
