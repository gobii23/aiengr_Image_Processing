# AI Image Processing - Pipe Removal

## Description
This project focuses on removing pipes from images using AI-based image processing techniques. It leverages image segmentation and inpainting techniques to detect and remove pipes while preserving the background.

## My Approach
1. **Preprocessing**: Convert the image to grayscale and apply edge detection.
2. **Pipe Detection**: Use morphological operations and contour detection to identify pipes.
3. **Mask Generation**: Create a mask of detected pipes to be removed.
4. **Image Inpainting**: Use OpenCV's inpainting techniques to reconstruct the background.

The script efficiently handles files by reading images using cv2.imread(), caching frequently accessed images with lru_cache, and writing processed images using cv2.imwrite() while ensuring the output directory exists with os.makedirs(). It processes multiple images by scanning an input directory using os.listdir(), applying AI-based pipeline removal, and saving results in an output directory. To enhance performance, the script utilizes multi-threading with ProcessPoolExecutor, allowing parallel execution of image processing tasks across multiple CPU cores. Several advanced techniques are incorporated, including adaptive thresholding for precise feature extraction, Canny edge detection for identifying pipeline lines, Hough Transform for detecting straight lines, connected component analysis for preserving symbols and text, morphological operations (dilation and filtering) for refining masks, and image inpainting to seamlessly remove unwanted lines while maintaining image integrity. If threading is enabled, the workload is distributed across available CPU threads, significantly improving processing speed. The main function initializes directories and calls batch_process_pid_diagrams() to handle images efficiently, making the script highly optimized for large-scale AI-driven image processing.

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
