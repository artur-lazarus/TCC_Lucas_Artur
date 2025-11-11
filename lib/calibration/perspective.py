import cv2, numpy as np

def _ord(pts):
    pts = np.array(pts, np.float32)
    xs = np.argsort(pts[:,0])
    l, r = pts[xs[:2]], pts[xs[2:]]
    tl, bl = l[np.argsort(l[:,1])]
    tr, br = r[np.argsort(r[:,1])]
    return np.array([tl, tr, br, bl], np.float32)

def unwarp(img, pts, interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_REPLICATE, border_value=0):
        """Apply a perspective transform to img, mapping pts to a rectangle.

        Params:
            - img: input image (H,W[,C])
            - pts: iterable of 4 (x,y) points in source image
            - interpolation: cv2 interpolation flag (default: INTER_LANCZOS4 for higher quality)
            - border_mode: border handling for areas outside input (default: REPLICATE)
            - border_value: used if border_mode is BORDER_CONSTANT
        """
        q = _ord(pts)
        # Compute output size based on the geometry of the quadrilateral
        w = int(max(np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[1] - q[0])))
        h = int(max(np.linalg.norm(q[1] - q[2]), np.linalg.norm(q[0] - q[3])))
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        M = cv2.getPerspectiveTransform(q, dst)
        return cv2.warpPerspective(img, M, (w, h), flags=interpolation, borderMode=border_mode, borderValue=border_value)
