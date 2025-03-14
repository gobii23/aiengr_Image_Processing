# AI-Powered Image Processing Project

## Approach
This project leverages AI-driven image processing and inference, utilizing various Python libraries for efficient image handling, model predictions, and visualization. The key steps involved are:

### 1. **Loading and Preprocessing Images**
   - Images are loaded and manipulated using `PIL` and `cv2`
   - `BytesIO` is used to handle image data in memory for seamless processing

### 2. **Model Inference**
   - The model is loaded via the `get_model` function from `inference.py`
   - Predictions are generated and visualized for analysis

### 3. **Visualization**
   - `matplotlib` and `supervision (sv)` are used for displaying processed images and annotation results

To enhance output efficiency, we curated datasets from Roboflow, mapping specific data points for training the model. The trained model is deployed using an API in Google Colab. Initially, it identifies symbols using a custom pre-trained model, followed by background removal to eliminate pipeline structures.

## How to Run the Code

1. **Install Dependencies**:
   ```bash
   pip install numpy opencv-python matplotlib requests pillow supervision
   ```

2. **Run the Jupyter Notebook**:
   - Open Jupyter Notebook:
     ```bash
     jupyter notebook aiengr.ipynb
     ```
   - Execute the cells sequentially to process images and obtain results

## Libraries Used
- **`numpy`** – Numerical operations
- **`opencv-python (cv2)`** – Image processing
- **`matplotlib`** – Data visualization
- **`requests`** – Fetching external data
- **`Pillow (PIL)`** – Image handling
- **`supervision (sv)`** – Annotation and drawing
- **`io.BytesIO`** – In-memory image operations
- **`inference.py`** – Custom model inference module

This streamlined approach ensures accurate symbol detection, efficient masking, and high-quality AI-driven image analysis.
