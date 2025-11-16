import os
import numpy as np
import cv2

def pixel_vp_to_cam_dir(vp_px, K):
    """(u,v) -> unit direction in camera coords via K^{-1}."""
    vp = np.array([vp_px[0], vp_px[1], 1.0], dtype=np.float64)
    d = np.linalg.inv(K) @ vp
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise ValueError("Vanishing point produced near-zero direction.")
    return d / n

def get_rotation_matrix_from_vps(vertical_vp_px, road_vp_px, K):
    """Return r1 (along road), r2 (across road), r3 (up)."""
    v = -pixel_vp_to_cam_dir(vertical_vp_px, K)
    x = pixel_vp_to_cam_dir(road_vp_px, K)

    # r3 = vertical (unit)
    r3 = v / np.linalg.norm(v)

    # r1 = road direction, orthogonalized wrt r3
    r1 = x - np.dot(x, r3) * r3
    n1 = np.linalg.norm(r1)
    if n1 < 1e-12:
        raise ValueError("road_vp is nearly collinear with vertical_vp.")
    r1 /= n1

    # r2 = r3 × r1 (right-handed basis)
    r2 = np.cross(r3, r1)
    r2 /= np.linalg.norm(r2)

    # Orthonormalize + enforce det(R)=+1
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
    r1, r2, r3 = R[:,0], R[:,1], R[:,2]
    return r1, r2, r3

def build_img_to_bird_homography(img_shape, K, r1, r2, scale=None, margin=0.02, roi_polygon=None, target_width_px=1280.0):
    """
    Returns M_img_to_bird (3x3) and (W_out, H_out).
    Uses H_img->plane = (K [r1 r2 -r3])^{-1}. Global scale is arbitrary;
    we set t ∝ -r3 which is valid since homographies are up to scale.
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    r1 /= np.linalg.norm(r1); r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2); r3 /= np.linalg.norm(r3)

    H_plane_to_img = K @ np.column_stack((r1, r2, -r3))  # t ∝ -r3
    H_img_to_plane = np.linalg.inv(H_plane_to_img)

    H_img, W_img = img_shape[:2]

    # Robust bounds via sampling; if roi_polygon provided, restrict sampling to ROI
    if roi_polygon is None:
        xs = np.linspace(0, W_img - 1, 25)
        ys = np.linspace(0, H_img - 1, 25)
        grid_pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    else:
        poly = np.asarray(roi_polygon, dtype=np.int32).reshape(-1, 1, 2)
        roi_mask = np.zeros((H_img, W_img), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [poly], 255)
        # pick an adaptive subset of ROI pixels (up to ~5000 samples)
        ys_idx, xs_idx = np.where(roi_mask > 0)
        if ys_idx.size == 0:
            raise ValueError("ROI polygon has zero area or lies outside image.")
        N = ys_idx.size
        max_samples = 5000
        if N > max_samples:
            stride = int(np.ceil(np.sqrt(N / max_samples)))
        else:
            stride = 1
        ys_idx = ys_idx[::stride]
        xs_idx = xs_idx[::stride]
        grid_pts = np.stack([xs_idx, ys_idx], axis=1).astype(np.float64)

    ones = np.ones((grid_pts.shape[0], 1), dtype=np.float64)
    grid_h = np.hstack([grid_pts, ones]).T  # 3xN

    plane_h = H_img_to_plane @ grid_h  # 3xN
    w = plane_h[2, :]
    good = np.abs(w) > 1e-6
    if not np.any(good):
        # Fallback to 4 corners
        corners_img = np.array([
            [0,        0,        1],
            [W_img-1., 0,        1],
            [W_img-1., H_img-1., 1],
            [0,        H_img-1., 1]
        ], dtype=np.float64).T  # 3x4
        corners_plane_h = H_img_to_plane @ corners_img
        w = corners_plane_h[2, :]
        good = np.abs(w) > 1e-6
        finite_pts = (corners_plane_h[:2, good] / w[good]).T
    else:
        finite_pts = (plane_h[:2, good] / w[good]).T

    if finite_pts.shape[0] < 3:
        raise RuntimeError("Homography produced <3 finite samples. Check vanishing points/intrinsics; horizon may pass inside image.")

    Xmin, Ymin = finite_pts.min(axis=0)
    Xmax, Ymax = finite_pts.max(axis=0)

    # expand a bit so we don't clip edges
    dx, dy = (Xmax - Xmin), (Ymax - Ymin)
    Xmin -= margin*dx; Xmax += margin*dx
    Ymin -= margin*dy; Ymax += margin*dy

    # choose scale (pixels per world-unit). If not given, aim for target_width_px (ROI-driven)
    if scale is None:
        scale = float(target_width_px) / max(1e-9, (Xmax - Xmin))

    W_out = int(np.clip(int(round((Xmax - Xmin) * scale)), 32, 8192))
    H_out = int(np.clip(np.ceil((Ymax - Ymin) * scale), 32, 8192))

    L_world_to_bird = np.array([
        [scale,    0.0,  -scale*Xmin],
        [0.0,    scale,  -scale*Ymin],
        [0.0,      0.0,            1]
    ], dtype=np.float64)

    M_img_to_bird = L_world_to_bird @ H_img_to_plane
    return M_img_to_bird, (W_out, H_out)

def f_from_two_orthogonal_vps(v1, v2, cx, cy):
    du1, dv1 = v1[0]-cx, v1[1]-cy
    du2, dv2 = v2[0]-cx, v2[1]-cy
    f2 = -(du1*du2 + dv1*dv2)
    if f2 <= 0:
        raise ValueError("Inconsistent VPs or assumptions (got f^2 <= 0).")
    return f2**0.5

def _debug_dump(image, K, r1, r2, H_img_to_plane, M_img_to_bird, out_size, out_dir, vps_px=None):
    """Save several debug visuals to help diagnose geometry issues."""
    os.makedirs(out_dir, exist_ok=True)
    H_img, W_img = image.shape[:2]

    # Numeric checks
    r1n, r2n = np.linalg.norm(r1), np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    r3n = np.linalg.norm(r3)
    R = np.column_stack([
        r1 / (r1n + 1e-12),
        r2 / (r2n + 1e-12),
        r3 / (r3n + 1e-12),
    ])
    ortho_err = np.linalg.norm(R.T @ R - np.eye(3))
    detR = np.linalg.det(R)

    print("[DEBUG] norms r1,r2,r3:", r1n, r2n, r3n)
    print("[DEBUG] ortho_err ||R^T R - I||:", ortho_err, " det(R):", detR)

    # Denominator stats (how close image rays are to being parallel to plane)
    xs = np.linspace(0, W_img - 1, 49)
    ys = np.linspace(0, H_img - 1, 49)
    grid_pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    grid_h = np.hstack([grid_pts, np.ones((grid_pts.shape[0],1))]).T
    plane_h = H_img_to_plane @ grid_h
    w = plane_h[2,:]
    print(f"[DEBUG] denom (n^T K^-1 p) stats -> min:{w.min():.3e} max:{w.max():.3e} median:{np.median(w):.3e} zero_like:{np.count_nonzero(np.abs(w)<1e-6)}")

    # Draw mapped image boundary on bird canvas
    W_out, H_out = out_size
    bird_dbg = np.full((H_out, W_out, 3), 20, dtype=np.uint8)
    img_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    def map_pts_to_bird(pts_xy):
        pts_h = np.hstack([pts_xy, np.ones((pts_xy.shape[0],1))]).T
        q = (M_img_to_bird @ pts_h).T
        good = np.abs(q[:,2]) > 1e-6
        out = np.zeros((pts_xy.shape[0],2), dtype=np.float64)
        out[good] = q[good,:2] / q[good,2:3]
        return out, good

    # Sample each image edge
    edges = [
        np.stack([np.linspace(0, W_img-1, 200), np.zeros(200)], axis=1),
        np.stack([np.full(200, W_img-1), np.linspace(0, H_img-1, 200)], axis=1),
        np.stack([np.linspace(W_img-1, 0, 200), np.full(200, H_img-1)], axis=1),
        np.stack([np.zeros(200), np.linspace(H_img-1, 0, 200)], axis=1)
    ]
    for e in edges:
        b, good = map_pts_to_bird(e)
        b = b[good]
        if b.shape[0] >= 2:
            b_int = np.int32(b.reshape(-1,1,2))
            cv2.polylines(bird_dbg, [b_int], isClosed=False, color=(0,255,0), thickness=2)

    # Draw vanishing points if provided and visible
    if vps_px is not None:
        for (name, vp) in vps_px:
            u, v = vp
            if 0 <= u < W_img and 0 <= v < H_img:
                cv2.circle(img_vis, (int(round(u)), int(round(v))), 6, (0,0,255), -1)
                cv2.putText(img_vis, name, (int(u)+6, int(v)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    # Save original and bird boundary debug
    cv2.imwrite(os.path.join(out_dir, "original.png"), img_vis)
    cv2.imwrite(os.path.join(out_dir, "bird_boundary.png"), bird_dbg)

def main():
    input_image_path = "dataset/session0_center/screen.png"
    # If you truly want grayscale, read as such:
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(input_image_path)

    # TODO: put your real intrinsics here (float64!)
    

    # TODO: your detected vanishing points in pixel coords
    vertical_vp_px = (-5379.92779268, 73377.21463969)
    road_vp_px     = (120.20467933, 18.06131721)
    #vertical_vp_px = (-434.0, 17851.0)
    #road_vp_px     = (163, 47)

    f=f_from_two_orthogonal_vps(vertical_vp_px, road_vp_px, 960.0, 540.0)
    K = np.array([
        [ f,   0.0, 960.0],
        [   0.0, f, 540.0],
        [   0.0,   0.0,   1.0]
    ], dtype=np.float64)

    r1, r2, r3 = get_rotation_matrix_from_vps(vertical_vp_px, road_vp_px, K)
    # Optional: only dewarp pixels inside a polygon (in image pixel coords)
    # Example: uncomment and edit the points below (clockwise or ccw)
    # polygon_pts = np.array([[100,600],[500,400],[1200,420],[900,700]], dtype=np.int32)
    polygon_pts = np.array([(315,165),(185,174),(612,1069), (1740,1072)], dtype=np.int32)

    M_img_to_bird, (W_out, H_out) = build_img_to_bird_homography(
        image.shape, K, r1, r2, scale=None, margin=0.01, roi_polygon=polygon_pts, target_width_px=1280.0
    )
    print(f"[DEBUG] bird output size: W_out={W_out}, H_out={H_out}")
    print("Matrix:", M_img_to_bird)
    # Optional: only dewarp pixels inside a polygon (in image pixel coords)
    # polygon_pts = np.array([[100,600],[500,400],[1200,420],[900,700]], dtype=np.int32)
    #polygon_pts = np.array([(9, 334), (755, 1072), (1909, 643), (724, 292)], dtype=np.int32)

    H_img, W_img = image.shape[:2]
    debug_dir = os.path.join("test_video", "vp_debug", "tmpframes")
    mask_src = np.full((H_img, W_img), 255, dtype=np.uint8)
    input_for_warp = image
    if polygon_pts is not None:
        poly = np.asarray(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
        mask_src[:] = 0
        cv2.fillPoly(mask_src, [poly], 255)
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "original_with_polygon.png"), overlay)
        input_for_warp = cv2.bitwise_and(image, image, mask=mask_src)
        cv2.imwrite(os.path.join(debug_dir, "masked_input.png"), input_for_warp)

    bird = cv2.warpPerspective(
        input_for_warp,
        M_img_to_bird,
        (W_out, H_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    # Recompute H_img_to_plane for debug drawing
    r1n = r1/np.linalg.norm(r1)
    r2n = r2/np.linalg.norm(r2)
    r3n = np.cross(r1n, r2n); r3n /= np.linalg.norm(r3n)
    H_img_to_plane = np.linalg.inv(K @ np.column_stack((r1n, r2n, -r3n)))

    debug_dir = os.path.join("test_video", "vp_debug", "tmpframes")

    _debug_dump(
        image,
        K,
        r1,
        r2,
        H_img_to_plane,
        M_img_to_bird,
        (W_out, H_out),
        debug_dir,
        vps_px=[("vertical_vp", vertical_vp_px), ("road_vp", road_vp_px)],
    )
    # Save bird image to disk as well
    cv2.imwrite(os.path.join(debug_dir, "bird.png"), bird)
    # Also create a coverage mask to visualize which output pixels are sourced from the input
    # (respecting polygon ROI if provided)
    if polygon_pts is None:
        mask_src = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
    else:
        mask_src = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        poly = np.asarray(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask_src, [poly], 255)
    bird_mask = cv2.warpPerspective(
        mask_src,
        M_img_to_bird,
        (W_out, H_out),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cv2.imwrite(os.path.join(debug_dir, "bird_mask.png"), bird_mask)

    # Auto-crop to the minimal rectangle containing valid coverage (optional view)
    ys, xs = np.where(bird_mask > 0)
    if ys.size > 0 and xs.size > 0:
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        bird_cropped = bird[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(debug_dir, "bird_cropped.png"), bird_cropped)
    cv2.imshow("Bird's Eye View", bird)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()