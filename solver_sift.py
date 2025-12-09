import cv2
import numpy as np
import math
from typing import Dict, Tuple, Optional, Any

import config
from jigsawlver import load_piece_images_from_folder, load_image

# ------------------------------------------------------------
# Algorithm 1: With Original Image (SIFT + RANSAC)
# ------------------------------------------------------------

class JigsawSolverWithOriginal:
    """
    Implements the 'Algorithm with the Original Image' (Section 2.1 in the Ref paper).
    
    Strategy:
      1. Detect SIFT features (keypoints & descriptors) in the original reference image.
      2. For each puzzle piece:
         a. Detect SIFT features.
         b. Match piece features to original features using KNN.
         c. Use RANSAC to find the best Affine Transformation (rotation + translation).
      3. Reconstruct the puzzle by warping pieces to their calculated positions.
    """

    def __init__(self, original_img: np.ndarray):
        self.original_shape = original_img.shape
        # Convert to grayscale for feature detection
        if original_img.ndim == 3:
            self.original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            self.original_gray = original_img
            
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        # Initialize Brute-Force Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Pre-compute features for the original image (reference)
        self.kp_orig, self.des_orig = self.sift.detectAndCompute(self.original_gray, None)

    def solve_and_reconstruct(self, pieces: Dict[int, np.ndarray]) -> Tuple[np.ndarray, Dict[int, Any]]:
        """
        Match all pieces and reconstruct the image on a blank canvas.
        
        Returns:
            canvas: The reconstructed image.
            transforms: Dict mapping piece_id to (Matrix, angle, center).
        """
        h, w = self.original_shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        transforms = {}

        for pid, img in pieces.items():
            # Find where this piece belongs
            match = self.match_piece(img)
            if match:
                M, angle_deg, (cx, cy) = match
                transforms[pid] = (M, angle_deg, (cx, cy))

                # Warp the piece to its correct location on the canvas
                warped = cv2.warpAffine(img, M, (w, h))
                
                # Overlay the warped piece (ignoring black background)
                mask = np.any(warped > 0, axis=2)
                canvas[mask] = warped[mask]
        
        return canvas, transforms

    def match_piece(self, piece_img: np.ndarray, ratio_thresh: float = 0.7, ransac_thresh: float = 3.0) -> Optional[Tuple[np.ndarray, float, Tuple[float, float]]]:
        """
        Matches a single piece to the original image.
        
        Steps:
        1. Detect features in piece.
        2. KNN Match with original features (k=2).
        3. Apply Lowe's Ratio Test to filter good matches.
        4. Use RANSAC to estimate Affine Transform (robust to outliers).
        """
        gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY) if piece_img.ndim == 3 else piece_img
        kp_piece, des_piece = self.sift.detectAndCompute(gray, None)
        
        # Need enough features to match
        if des_piece is None or len(kp_piece) < 3 or self.des_orig is None or len(self.des_orig) < 3: return None

        # KNN Match: Find 2 best matches for each descriptor
        good = [m for m, n in self.bf.knnMatch(des_piece, self.des_orig, k=2) if m.distance < ratio_thresh * n.distance]
        if len(good) < 3: return None

        # Extract matched point coordinates
        pts_piece = np.float32([kp_piece[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_orig = np.float32([self.kp_orig[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Estimate Affine Transform (2D rotation + translation + scale)
        M, _ = cv2.estimateAffinePartial2D(pts_piece, pts_orig, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if M is None: return None

        # Calculate center position and rotation angle from the matrix
        h, w = gray.shape[:2]
        center_orig = M @ np.array([[w / 2.0, h / 2.0, 1.0]], dtype=np.float32).T
        return M, math.degrees(math.atan2(M[1, 0], M[0, 0])), (float(center_orig[0, 0]), float(center_orig[1, 0]))

    def solve(self, pieces: Dict[int, np.ndarray]) -> Dict[int, Tuple[float, Tuple[float, float]]]:
        """
        Wrapper to just get the transformation data without reconstruction.
        """
        results = {}
        for pid, img in pieces.items():
            m = self.match_piece(img)
            if m is None:
                continue
            M, angle_deg, (cx, cy) = m
            results[pid] = (angle_deg, (cx, cy))
        return results


def solve_with_original(original_path: str, pieces_folder: str) -> Dict[int, Tuple[float, Tuple[float, float]]]:
    """Helper function to load images and run the solver."""
    orig = load_image(original_path)
    if orig is None:
        raise ValueError(f"Could not load original image: {original_path}")
    pieces = load_piece_images_from_folder(pieces_folder)
    solver = JigsawSolverWithOriginal(orig)
    return solver.solve(pieces)
