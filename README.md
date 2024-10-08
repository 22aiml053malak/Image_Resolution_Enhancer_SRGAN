# Image_Resolution_Enhancer_SRGAN
This project uses a Super-Resolution Generative Adversarial Network (SRGAN) to enhance the resolution of images by 4x. Users can upload low-resolution images, and the web application—built using Flask—processes these images with a pre-trained SRGAN model, returning high-quality images. 


## Features
- **Image Upload:** Users can upload low-resolution images for processing.
- **4x Enhancement:** Utilizes a pre-trained SRGAN model to improve image resolution.
- **Download Option:** Users can download the enhanced image.
- **Flask Web App:** A simple interface for image enhancement.

## Technology Stack
- **SRGAN:** Pre-trained SRGAN model for super-resolution.
- **Flask:** Backend web framework for handling image uploads and processing.
- **HTML/CSS/JavaScript:** Basic frontend for user interaction.

## Installation

### Prerequisites
- Python 3.7+
- Flask
- Pre-trained SRGAN model

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/srgan-image-enhancer.git
    cd srgan-image-enhancer
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask application:
    ```bash
    python app.py
    ```

4. Open your web browser and go to:
    ```
    http://127.0.0.1:5000/
    ```

## How to Use

1. Upload a low-resolution image.
2. Wait for the model to process the image.
3. Download the high-resolution image.

## Example Output
- Input: Low-resolution image (e.g., 64x64)
- Output: Enhanced high-resolution image (e.g., 256x256)
