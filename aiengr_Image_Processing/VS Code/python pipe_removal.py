import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import lru_cache
import time

def remove_pid_pipeline_lines(image_path, output_path=None, show_results=True):
    """
    Specialized function to remove horizontal and vertical pipeline lines in P&ID diagrams
    while preserving symbols, text, and other components.
    
    Args:
        image_path (str): Path to the input P&ID diagram image
        output_path (str, optional): Path to save the output image
        show_results (bool): Whether to display the results
        
    Returns:
        numpy.ndarray: Processed image with pipeline lines removed
        float: Processing time in seconds
    """
    # Start timing
    start_time = time.time()
    
    # Read the image - use IMREAD_UNCHANGED to handle different image types correctly
    original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale if needed, but avoid unnecessary conversion
    if len(original.shape) == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original.copy()
    
    # Step 1: Apply preprocessing to enhance line detection
    # Use bilateral filter to preserve edges while reducing noise (better for P&ID diagrams)
    filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    
    # Step 2: Apply adaptive thresholding optimized for P&ID diagrams
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Step 3: Targeted edge detection for P&ID pipeline lines
    edges = cv2.Canny(filtered, 50, 150, apertureSize=3)
    
    # Step 4: Use probabilistic Hough Transform with parameters tuned for pipelines
    # Pipeline lines in P&ID are typically longer, straighter lines
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,  # Lower threshold to catch pipeline lines
        minLineLength=40,  # Longer minimum length for pipeline lines
        maxLineGap=15     # Larger gap for potentially broken pipeline lines
    )
    
    # Create a mask for the pipeline lines to remove
    pipeline_mask = np.zeros_like(gray, dtype=np.uint8)
    
    if lines is not None and len(lines) > 0:
        # Filter lines by angle and length to identify pipelines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate length and angle
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Identify horizontal or vertical pipeline lines with tighter tolerances
            # P&ID pipelines are typically very straight
            is_horizontal = angle < 2 or angle > 178
            is_vertical = 88 < angle < 92
            
            # Additional filter for minimum line length for pipelines (adjust as needed)
            is_pipeline = length > 50
            
            if (is_horizontal or is_vertical) and is_pipeline:
                # Draw line with thickness proportional to pipeline thickness
                thickness = 3  # Typical P&ID pipeline line thickness
                cv2.line(pipeline_mask, (x1, y1), (x2, y2), 255, thickness)
    
    # Step 5: Identify and protect symbols, text, and components using connected components
    # P&ID symbols need special care to be preserved
    nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
    
    # Create a mask for symbols and text components
    symbol_mask = np.zeros_like(binary, dtype=np.uint8)
    
    # P&ID symbols typically have specific characteristics
    # Skip background component (index 0)
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Identify potential symbols and text
        # - Symbols in P&ID are usually compact with particular width/height ratios
        # - Text has medium area but not too elongated
        is_symbol = (area > 30 and area < 3000) and (0.5 < width/height < 2.0)
        is_text = (area > 20 and area < 1000) and max(width, height) < 100
        
        if is_symbol or is_text:
            component_mask = (labels == i).astype(np.uint8) * 255
            symbol_mask = cv2.bitwise_or(symbol_mask, component_mask)
    
    # Step 6: Dilate symbol mask to protect borders of symbols and text
    # P&ID symbols often need more protection to avoid damage
    symbol_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    protected_areas = cv2.dilate(symbol_mask, symbol_kernel, iterations=2)
    
    # Step 7: Refine pipeline mask to avoid removing protected areas
    refined_pipeline_mask = cv2.bitwise_and(pipeline_mask, cv2.bitwise_not(protected_areas))
    
    # Step 8: Dilate the pipeline mask slightly to ensure complete removal
    pipeline_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inpaint_mask = cv2.dilate(refined_pipeline_mask, pipeline_kernel, iterations=1)
    
    # Step 9: Apply inpainting to remove the pipeline lines
    result = cv2.inpaint(gray, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Convert back to original color format if needed
    if len(original.shape) == 3:
        result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        # Handle alpha channel if present
        if original.shape[2] == 4:
            alpha = original[:, :, 3]
            result_bgra = cv2.cvtColor(result_color, cv2.COLOR_BGR2BGRA)
            result_bgra[:, :, 3] = alpha
            result_color = result_bgra
    else:
        result_color = result
    
    # Calculate and print processing time
    processing_time = time.time() - start_time
    print(f"Pipeline removal completed in {processing_time:.2f} seconds")
    
    # Visualize results if requested
    if show_results:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.title('Original P&ID Diagram')
        if len(original.shape) == 3 and original.shape[2] == 3:
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(gray, cmap='gray')
        plt.axis('off')
        
        plt.subplot(232)
        plt.title('Detected Edges')
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        
        plt.subplot(233)
        plt.title('Pipeline Line Mask')
        plt.imshow(pipeline_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(234)
        plt.title('Protected Symbols & Text')
        plt.imshow(protected_areas, cmap='gray')
        plt.axis('off')
        
        plt.subplot(235)
        plt.title('Final Inpaint Mask')
        plt.imshow(inpaint_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(236)
        plt.title(f'P&ID without Pipelines ({processing_time:.2f}s)')
        if len(result_color.shape) == 3 and result_color.shape[2] == 3:
            plt.imshow(cv2.cvtColor(result_color, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(result_color, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Save the results if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_color)
    
    return result_color, processing_time

@lru_cache(maxsize=32)
def cached_read_image(image_path):
    """Cache image reading to improve performance when the same image is processed multiple times"""
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

def batch_process_pid_diagrams(input_dir, output_dir, show_results=False, parallel=False):
    """
    Process multiple P&ID diagram images with optional parallel processing
    
    Args:
        input_dir (str): Directory containing input P&ID diagrams
        output_dir (str): Directory to save processed diagrams
        show_results (bool): Whether to display results
        parallel (bool): Whether to use parallel processing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"No P&ID diagram files found in {input_dir}. Please add images and try again.")
        return
    
    total_start_time = time.time()
    processing_times = []
    
    if parallel and cv2.getNumberOfCPUs() > 1:
        # Parallel processing using multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        
        # Create a partial function with fixed parameters
        process_func = partial(
            process_single_pid_diagram, 
            input_dir=input_dir, 
            output_dir=output_dir, 
            show_results=False  # Don't show results in parallel mode
        )
        
        # Use up to number of CPUs - 1 workers
        num_workers = max(1, cv2.getNumberOfCPUs() - 1)
        print(f"Using {num_workers} parallel workers for P&ID processing")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_func, image_files))
        
        # Collect results
        for img_file, success, proc_time in results:
            if success:
                processing_times.append(proc_time)
                print(f"Successfully processed P&ID: {img_file} in {proc_time:.2f} seconds")
            else:
                print(f"Error processing P&ID: {img_file}")
    else:
        # Sequential processing
        for img_file in image_files:
            success, proc_time = process_single_pid_diagram(
                img_file, input_dir, output_dir, show_results
            )
            if success:
                processing_times.append(proc_time)
    
    # Calculate overall statistics
    total_time = time.time() - total_start_time
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    print(f"\nP&ID Processing Summary:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average per P&ID diagram: {avg_time:.2f} seconds")
    print(f"Processed {len(processing_times)} of {len(image_files)} P&ID diagrams successfully")

def process_single_pid_diagram(img_file, input_dir, output_dir, show_results):
    """Process a single P&ID diagram and return success status and processing time"""
    input_path = os.path.join(input_dir, img_file)
    output_path = os.path.join(output_dir, f"no_pipelines_{img_file}")
    
    try:
        # Process the image and get processing time
        _, proc_time = remove_pid_pipeline_lines(
            input_path, output_path, show_results=show_results
        )
        return img_file, True, proc_time
    except Exception as e:
        print(f"Error processing P&ID diagram {img_file}: {str(e)}")
        return img_file, False, 0

def main():
    """
    Main function for P&ID pipeline removal
    """
    # Define directories
    input_dir = "input_pid_diagrams"
    output_dir = "processed_pid_results"
    
    # Create directories if they don't exist
    for directory in [input_dir, output_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Processing P&ID diagrams from {input_dir}...")
    print(f"Results will be saved to {output_dir}")
    
    # Check if we can use parallel processing
    use_parallel = cv2.getNumberOfCPUs() > 1
    
    # Process all P&ID diagrams
    batch_process_pid_diagrams(input_dir, output_dir, show_results=True, parallel=use_parallel)

if __name__ == "__main__":
    main()