#!/usr/bin/env python
"""
compute_vp_w_from_homography.py

Uses the existing functions in homography.py to compute the vertical vanishing point
from two horizontal vanishing points (vp_u and vp_v).
"""

import sys
import os
import numpy as np
import cv2
import logging

# Add parent directory to path to import homography
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from homography import pixel_vp_to_cam_dir, f_from_two_orthogonal_vps

def compute_vp_w_using_cross_product(vp_u, vp_v, image_shape):
    """
    Compute vertical vanishing point from two horizontal VPs using homography.py functions.
    
    Args:
        vp_u: Vanishing point in road direction (x, y)
        vp_v: Vanishing point perpendicular to road (x, y)  
        image_shape: Tuple (height, width) for the image
        
    Returns:
        vp_w: Vertical vanishing point (x, y)
        K: Camera intrinsic matrix
    """
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    
    # Estimate focal length using the existing function from homography.py
    f = f_from_two_orthogonal_vps(vp_u, vp_v, cx, cy)
    
    K = np.array([
        [f,   0.0, cx],
        [0.0, f,   cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    logging.info(f"Estimated camera intrinsics:")
    logging.info(f"  Focal length: {f:.2f} px")
    logging.info(f"  Principal point: ({cx:.2f}, {cy:.2f})")
    
    # Convert VPs to camera directions using homography.py function
    d_u = pixel_vp_to_cam_dir(vp_u, K)
    d_v = pixel_vp_to_cam_dir(vp_v, K)
    
    # Verify orthogonality
    dot_product = np.dot(d_u, d_v)
    logging.info(f"Orthogonality check: d_u · d_v = {dot_product:.6f}")
    
    # Compute vertical direction as cross product
    d_w = np.cross(d_u, d_v)
    d_w = d_w / np.linalg.norm(d_w)
    
    # Project back to image plane to get vp_w
    vp_w_homogeneous = K @ d_w
    vp_w = vp_w_homogeneous[:2] / vp_w_homogeneous[2]
    
    # Verify mutual orthogonality
    logging.info(f"Verification: d_u · d_w = {np.dot(d_u, d_w):.6f}, d_v · d_w = {np.dot(d_v, d_w):.6f}")
    
    return vp_w, K

def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load existing VPs
    vp_u = np.load(os.path.join(THIS_DIR, "vp_u.npy"))
    vp_v = np.load(os.path.join(THIS_DIR, "vp_v.npy"))
    
    logging.info(f"Loaded VP-u (road): {vp_u}")
    logging.info(f"Loaded VP-v (perpendicular): {vp_v}")
    
    # Get image shape from video
    video_path = "/Users/lucmarts/Documents/Pessoal/TCC_Lucas_Artur/assets/video.avi"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logging.error("Could not read frame from video")
        return
    
    # Compute vp_w
    logging.info("\nComputing VP-w (vertical) using homography.py functions...")
    vp_w, K = compute_vp_w_using_cross_product(vp_u, vp_v, frame.shape)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Vertical Vanishing Point (VP-w): {vp_w}")
    logging.info(f"{'='*60}\n")
    
    # Save results
    np.save(os.path.join(THIS_DIR, "vp_w.npy"), vp_w)
    logging.info(f"Saved to vp_w.npy")
    
    np.savez(os.path.join(THIS_DIR, "camera_intrinsics_from_vps.npz"),
             K=K, vp_u=vp_u, vp_v=vp_v, vp_w=vp_w)
    logging.info(f"Saved camera intrinsics to camera_intrinsics_from_vps.npz")

if __name__ == "__main__":
    main()
