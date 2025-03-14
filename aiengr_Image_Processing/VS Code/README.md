# AI Image Processing - Pipe Removal

## Description
This project focuses on removing pipes from images using AI-based image processing techniques. It leverages image segmentation and inpainting techniques to detect and remove pipes while preserving the background.

## My Approach
1. **Preprocessing**: Convert the image to grayscale and apply edge detection.
2. **Pipe Detection**: Use morphological operations and contour detection to identify pipes.
3. **Mask Generation**: Create a mask of detected pipes to be removed.
4. **Image Inpainting**: Use OpenCV's inpainting techniques (e.g., Telea and Navier-Stokes methods) to reconstruct the background.

## How to Run the Code

### Prerequisites
Ensure you have Python installed along with the required libraries.

### Installation
Install the necessary dependencies:
```bash
pip install opencv-python numpy matplotlib
```

### Running the Script
Execute the script with an image as input:
```bash
python pipe_removal.py --input path/to/image.jpg --output path/to/output.jpg
```

## Libraries Used
- `opencv-python`: For image processing and inpainting.
- `numpy`: For numerical operations.
- `matplotlib`: For displaying results.

## Results
Before-and-after images are stored in the `results/` folder.
