import cv2
import numpy as np


def get_polygon_vertices(mask_path):
    """
    Extract vertex coordinates from a polygon mask in a PNG image.
    
    Args:
        mask_path (str): Path to the PNG mask file
        
    Returns:
        numpy.ndarray: Array of vertex coordinates [[x1, y1], [x2, y2], ...]
    """
    # Read the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise ValueError(f"Could not read mask from {mask_path}")
    
    # Threshold the mask to ensure binary image
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the mask (use CHAIN_APPROX_NONE for all points)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        raise ValueError("No contours found in the mask")
    
    # Get the largest contour (assuming it's the polygon)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Use a smaller epsilon for more precise approximation
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    vertices = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Reshape to get clean coordinate array
    vertices = vertices.reshape(-1, 2)
    
    return vertices


def main():
    # Example usage
    mask_path = "dataset/session0_left/video_mask.png"
    
    try:
        vertices = get_polygon_vertices(mask_path)
        print("Polygon vertices:")
        print(vertices)
        print(f"\nNumber of vertices: {len(vertices)}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
