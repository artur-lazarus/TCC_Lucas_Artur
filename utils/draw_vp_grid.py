#!/usr/bin/env python3
"""
Vanishing Point Grid Drawing Utility

This script draws a perspective grid on an image using two vanishing points (VPs).
Lines are drawn from each VP through evenly-spaced points on the image perimeter.

Usage:
    python draw_vp_grid.py <image_path> <vpu_x> <vpu_y> <vpv_x> <vpv_y> [options]
    
Examples:
    # Draw grid with default parameters
    python draw_vp_grid.py image.jpg 640 100 320 900
    
    # Draw grid with custom number of lines
    python draw_vp_grid.py image.jpg 640 100 320 900 --num-angles-u 12 --num-angles-v 8
    
    # Custom colors (BGR format)
    python draw_vp_grid.py image.jpg 640 100 320 900 --color-u 255,0,0 --color-v 0,255,0
    
    # Custom output file
    python draw_vp_grid.py image.jpg 640 100 320 900 -o output_grid.jpg
"""

import cv2
import numpy as np
import argparse
import os
import sys


def draw_line_from_vp(image, vp, angle_deg, color=(0, 255, 0), thickness=3):
    """
    Draw an infinite line from a vanishing point at a specified angle.
    The line is clipped to image boundaries to appear as if it extends infinitely.
    
    Args:
        image (np.ndarray): Input image
        vp (tuple): Vanishing point coordinates (x, y)
        angle_deg (float): Angle in degrees from horizontal
        color (tuple): Line color in BGR format
        thickness (int): Line thickness
    """
    h, w = image.shape[:2]
    vp_x, vp_y = vp
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Calculate direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Avoid division by zero
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return
    
    # Find intersections with image boundaries
    intersections = []
    
    # Left edge (x = 0)
    if abs(dx) > 1e-10:
        t = (0 - vp_x) / dx
        y = vp_y + t * dy
        if 0 <= y <= h:
            intersections.append((0, int(y)))
    
    # Right edge (x = w)
    if abs(dx) > 1e-10:
        t = (w - vp_x) / dx
        y = vp_y + t * dy
        if 0 <= y <= h:
            intersections.append((w - 1, int(y)))
    
    # Top edge (y = 0)
    if abs(dy) > 1e-10:
        t = (0 - vp_y) / dy
        x = vp_x + t * dx
        if 0 <= x <= w:
            intersections.append((int(x), 0))
    
    # Bottom edge (y = h)
    if abs(dy) > 1e-10:
        t = (h - vp_y) / dy
        x = vp_x + t * dx
        if 0 <= x <= w:
            intersections.append((int(x), h - 1))
    
    # Remove duplicate points (can happen at corners)
    unique_intersections = []
    for pt in intersections:
        if not any(abs(pt[0] - p[0]) < 2 and abs(pt[1] - p[1]) < 2 for p in unique_intersections):
            unique_intersections.append(pt)
    
    # Draw line between the two intersection points
    if len(unique_intersections) >= 2:
        cv2.line(image, unique_intersections[0], unique_intersections[1], color, thickness, cv2.LINE_AA)


def draw_vp_grid(image, vpu, vpv, num_angles_u=8, num_angles_v=8, 
                 color_u=(0, 255, 0), color_v=(255, 0, 0),
                 thickness=3, mark_vps=True, show_angles=False):
    """
    Draw a perspective grid on an image using two vanishing points.
    Lines are drawn from VPs through evenly-spaced points on the image perimeter.
    
    Args:
        image (np.ndarray or str): Input image (array or path)
        vpu (tuple): First vanishing point coordinates (x, y)
        vpv (tuple): Second vanishing point coordinates (x, y)
        num_angles_u (int): Number of lines from vpu
        num_angles_v (int): Number of lines from vpv
        color_u (tuple): Color for lines from vpu (BGR format)
        color_v (tuple): Color for lines from vpv (BGR format)
        thickness (int): Line thickness (default: 3)
        mark_vps (bool): Whether to mark vanishing points with circles
        show_angles (bool): Whether to show angle values on lines
    
    Returns:
        np.ndarray: Image with grid drawn
    """
    # Load image if path is provided
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Could not load image: {image}")
    
    # Make a copy to avoid modifying original
    result = image.copy()
    h, w = result.shape[:2]
    
    def get_bottom_points(points_per_side=10):
        """Generate evenly-spaced points on bottom edge (auxiliary only)"""
        points = []
        for i in range(points_per_side):
            x = int((i + 0.5) * w / points_per_side)
            points.append((x, h - 1))
        return points
    
    def get_right_points(points_per_side=10):
        """Generate evenly-spaced points on right edge (auxiliary only)"""
        points = []
        for i in range(points_per_side):
            y = int((i + 0.5) * h / points_per_side)
            points.append((w - 1, y))
        return points
    
    # Draw lines from vpu through bottom edge points only
    print(f"Drawing lines from VPu at {vpu} (10 points on bottom edge)...")
    perimeter_points_u = get_bottom_points(10)
    for pt in perimeter_points_u:
        # Calculate direction from VP to perimeter point
        dx = pt[0] - vpu[0]
        dy = pt[1] - vpu[1]
        
        # Skip if point coincides with VP
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            continue
        
        # Calculate angle in degrees
        angle_deg = np.degrees(np.arctan2(dy, dx))
        
        # Draw line from VP through this perimeter point
        draw_line_from_vp(result, vpu, angle_deg, color_u, thickness)
    
    # Draw lines from vpv through right edge points only
    print(f"Drawing lines from VPv at {vpv} (10 points on right edge)...")
    perimeter_points_v = get_right_points(10)
    for pt in perimeter_points_v:
        # Calculate direction from VP to perimeter point
        dx = pt[0] - vpv[0]
        dy = pt[1] - vpv[1]
        
        # Skip if point coincides with VP
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            continue
        
        # Calculate angle in degrees
        angle_deg = np.degrees(np.arctan2(dy, dx))
        
        # Draw line from VP through this perimeter point
        draw_line_from_vp(result, vpv, angle_deg, color_v, thickness)
    
    # Mark vanishing points (only if within reasonable range)
    if mark_vps:
        margin = max(w, h) * 2
        
        if -margin < vpu[0] < w + margin and -margin < vpu[1] < h + margin:
            cv2.circle(result, (int(vpu[0]), int(vpu[1])), 8, color_u, -1)
            cv2.circle(result, (int(vpu[0]), int(vpu[1])), 10, (255, 255, 255), 2)
            cv2.putText(result, "VPu", (int(vpu[0]) + 15, int(vpu[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_u, 2)
        
        if -margin < vpv[0] < w + margin and -margin < vpv[1] < h + margin:
            cv2.circle(result, (int(vpv[0]), int(vpv[1])), 8, color_v, -1)
            cv2.circle(result, (int(vpv[0]), int(vpv[1])), 10, (255, 255, 255), 2)
            cv2.putText(result, "VPv", (int(vpv[0]) + 15, int(vpv[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_v, 2)
    
    return result


def draw_radial_grid(image, vpu, vpv, num_radial=8, num_angular=12,
                     color_radial=(0, 255, 0), color_angular=(255, 0, 0),
                     thickness=3):
    """
    Draw a radial grid from a central point with angular divisions.
    
    Args:
        image (np.ndarray or str): Input image (array or path)
        vpu (tuple): Center point for radial lines (x, y)
        vpv (tuple): Reference point for orientation (x, y)
        num_radial (int): Number of radial lines
        num_angular (int): Number of angular divisions
        color_radial (tuple): Color for radial lines (BGR format)
        color_angular (tuple): Color for angular lines (BGR format)
        thickness (int): Line thickness
    
    Returns:
        np.ndarray: Image with radial grid drawn
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image)
    
    result = image.copy()
    h, w = result.shape[:2]
    
    # Calculate maximum radius (diagonal of image)
    max_radius = np.sqrt(w**2 + h**2)
    
    # Draw radial lines
    angle_step = 360.0 / num_radial
    for i in range(num_radial):
        angle = i * angle_step
        draw_line_from_vp(result, vpu, angle, color_radial, thickness)
    
    # Draw concentric circles (angular divisions)
    radius_step = max_radius / (num_angular + 1)
    for i in range(1, num_angular + 1):
        radius = int(radius_step * i)
        cv2.circle(result, (int(vpu[0]), int(vpu[1])), radius, color_angular, thickness)
    
    # Mark center point
    cv2.circle(result, (int(vpu[0]), int(vpu[1])), 8, (255, 255, 255), -1)
    cv2.circle(result, (int(vpu[0]), int(vpu[1])), 10, (0, 0, 0), 2)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Draw perspective grid on image using vanishing points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with two vanishing points
  python draw_vp_grid.py image.jpg 640 100 320 900
  
  # Custom number of lines per VP
  python draw_vp_grid.py image.jpg 640 100 320 900 --num-angles-u 12 --num-angles-v 8
  
  # Custom colors (BGR format)
  python draw_vp_grid.py image.jpg 640 100 320 900 --color-u 255,0,0 --color-v 0,255,0
  
  # Radial grid mode
  python draw_vp_grid.py image.jpg 640 480 320 240 --mode radial --num-radial 16
  
  # Show angle labels
  python draw_vp_grid.py image.jpg 640 100 320 900 --show-angles
        """
    )
    
    parser.add_argument('image', help='Input image path')
    parser.add_argument('vpu_x', type=float, help='VPu X coordinate')
    parser.add_argument('vpu_y', type=float, help='VPu Y coordinate')
    parser.add_argument('vpv_x', type=float, help='VPv X coordinate')
    parser.add_argument('vpv_y', type=float, help='VPv Y coordinate')
    
    parser.add_argument('--mode', choices=['standard', 'radial'], default='standard',
                       help='Grid drawing mode (default: standard)')
    
    parser.add_argument('--num-angles-u', type=int, default=8,
                       help='Number of lines from VPu (default: 8)')
    parser.add_argument('--num-angles-v', type=int, default=8,
                       help='Number of lines from VPv (default: 8)')
    
    parser.add_argument('--num-radial', type=int, default=8,
                       help='Number of radial lines (radial mode only, default: 8)')
    parser.add_argument('--num-angular', type=int, default=12,
                       help='Number of angular divisions (radial mode only, default: 12)')
    
    parser.add_argument('--color-u', type=str, default='0,255,0',
                       help='Color for VPu lines in BGR format (default: 0,255,0)')
    parser.add_argument('--color-v', type=str, default='255,0,0',
                       help='Color for VPv lines in BGR format (default: 255,0,0)')
    
    parser.add_argument('--thickness', type=int, default=3,
                       help='Line thickness (default: 3)')
    
    parser.add_argument('--no-mark-vps', action='store_true',
                       help='Do not mark vanishing points with circles')
    
    parser.add_argument('--show-angles', action='store_true',
                       help='Show angle values on lines')
    
    parser.add_argument('-o', '--output', type=str,
                       help='Output image path (default: input_grid.jpg)')
    
    parser.add_argument('--display', action='store_true',
                       help='Display the result in a window')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Parse colors
    try:
        color_u = tuple(map(int, args.color_u.split(',')))
        color_v = tuple(map(int, args.color_v.split(',')))
        
        if len(color_u) != 3 or len(color_v) != 3:
            raise ValueError("Colors must have 3 components (BGR)")
    except:
        print("Error: Invalid color format. Use BGR format like: 255,0,0")
        sys.exit(1)
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        sys.exit(1)
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Define vanishing points
    vpu = (args.vpu_x, args.vpu_y)
    vpv = (args.vpv_x, args.vpv_y)
    
    print(f"VPu: {vpu}")
    print(f"VPv: {vpv}")
    
    # Draw grid
    if args.mode == 'radial':
        print("Drawing radial grid...")
        result = draw_radial_grid(
            image, vpu, vpv,
            num_radial=args.num_radial,
            num_angular=args.num_angular,
            color_radial=color_u,
            color_angular=color_v,
            thickness=args.thickness
        )
    else:
        print("Drawing standard grid...")
        result = draw_vp_grid(
            image, vpu, vpv,
            num_angles_u=args.num_angles_u,
            num_angles_v=args.num_angles_v,
            color_u=color_u,
            color_v=color_v,
            thickness=args.thickness,
            mark_vps=not args.no_mark_vps,
            show_angles=args.show_angles
        )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.image)[0]
        output_path = f"{base_name}_grid.jpg"
    
    # Save result
    print(f"Saving result to: {output_path}")
    cv2.imwrite(output_path, result)
    print("✓ Grid drawing completed successfully!")
    
    # Display if requested
    if args.display:
        print("Displaying result (press any key to close)...")
        cv2.imshow('VP Grid', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image = cv2.imread('assets/rasp_frame.png')
    vpu = (1011.42, -17.03)
    vpv = (-81683.08, 803.08)
    
    result = draw_vp_grid(
        image, vpu, vpv,
        num_angles_u=8,
        num_angles_v=8,
        color_u=(0, 0, 255),
        color_v=(0, 255, 0)
    )

    cv2.imwrite('eldorado/output_grid.png', result)
