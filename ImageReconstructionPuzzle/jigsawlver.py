import os
import glob
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image

import config


# ------------------------------------------------------------
# Utility dataclasses
# ------------------------------------------------------------

@dataclass
class PiecePlacement:
    """
    Represents the final position and orientation of a single puzzle piece.
    
    Attributes:
        piece_id (int): The unique identifier of the piece (index in the loading list).
        orientation (int): The rotation applied to the piece (0, 1, 2, 3 for 0, 90, 180, 270 degrees CCW).
        row (int): The row index in the solved grid (0-based).
        col (int): The column index in the solved grid (0-based).
    """
    piece_id: int
    orientation: int
    row: int
    col: int


@dataclass
class PuzzleResult:
    """
    Container for the complete solution of the puzzle.
    
    Attributes:
        rows (int): Total number of rows in the solved puzzle.
        cols (int): Total number of columns in the solved puzzle.
        placements (List[PiecePlacement]): A list containing the placement details for every piece.
    """
    rows: int
    cols: int
    placements: List[PiecePlacement]


# ------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------

def load_image(path: str) -> Optional[np.ndarray]:
    """
    Robustly loads an image from a file path.
    
    Tries to use OpenCV first. If that fails (e.g., due to file format issues),
    it falls back to Pillow (PIL) and converts the result to an OpenCV-compatible BGR array.
    
    Args:
        path (str): The absolute or relative path to the image file.
        
    Returns:
        Optional[np.ndarray]: The loaded image as a NumPy array (BGR), or None if loading failed.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None: return img
    try:
        # Fallback: Use PIL to open, convert to RGB, then to BGR for OpenCV compatibility
        return cv2.cvtColor(np.array(Image.open(path).convert('RGB')), cv2.COLOR_RGB2BGR)
    except Exception: return None

def load_piece_images_from_folder(folder: str) -> Dict[int, np.ndarray]:
    """
    Loads all image files from a specified directory.
    
    This function scans the folder, sorts files alphabetically to ensure consistent ID assignment,
    and loads each valid image.
    
    Args:
        folder (str): Path to the directory containing puzzle piece images.
        
    Returns:
        Dict[int, np.ndarray]: A dictionary mapping piece ID (int) to image data (np.ndarray).
    
    Raises:
        ValueError: If no valid images are found in the folder.
    """
    pieces = {i: img for i, path in enumerate(sorted(glob.glob(os.path.join(folder, "*.*")))) if (img := load_image(path)) is not None}
    if not pieces: raise ValueError(f"No images loaded from {folder}")
    return pieces


def rotate_image_90(img: np.ndarray, k: int) -> np.ndarray:
    """
    Rotates an image by 90 degrees counter-clockwise, k times.
    
    Args:
        img (np.ndarray): The source image.
        k (int): The number of 90-degree rotations (0, 1, 2, 3, etc.).
        
    Returns:
        np.ndarray: The rotated image copy.
    """
    k = k % 4
    if k == 0:
        return img
    return np.rot90(img, k).copy()


def ensure_square_and_same_size(pieces: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Standardizes all pieces to be square and of the same size.
    
    This is crucial for the geometric solver (without original), which relies on
    consistent edge lengths for comparison. It resizes all pieces to the maximum
    dimension found in the first piece.
    
    Args:
        pieces (Dict[int, np.ndarray]): The raw loaded pieces.
        
    Returns:
        Dict[int, np.ndarray]: A new dictionary with resized, square pieces.
    """
    squared = {}
    if not pieces:
        return {}

    # Determine target size (max dimension of the first piece to preserve detail)
    first_img = next(iter(pieces.values()))
    h, w = first_img.shape[:2]
    target_side = max(h, w)

    for pid, img in pieces.items():
        # Resize to target_side x target_side
        # This distorts aspect ratio but keeps all pixels (interpolated)
        resized = cv2.resize(img, (target_side, target_side), interpolation=cv2.INTER_AREA)
        squared[pid] = resized
        
    return squared


def visualize_puzzle_result(result: PuzzleResult, pieces: Dict[int, np.ndarray], tile_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Reconstructs the solved puzzle into a single image for display.
    
    It creates a blank canvas and pastes each piece into its calculated position
    based on the PuzzleResult.
    
    Args:
        result (PuzzleResult): The solution containing piece positions and orientations.
        pieces (Dict[int, np.ndarray]): The original piece images.
        tile_size (Tuple[int, int], optional): Forced size for each tile (height, width).
                                               If None, infers from the first piece.
                                               
    Returns:
        np.ndarray: The full reconstructed puzzle image.
    """
    if not pieces:
        return np.zeros((100, 100, 3), dtype=np.uint8)
        
    if tile_size is None:
        first_piece = next(iter(pieces.values()))
        tile_h, tile_w = first_piece.shape[:2]
    else:
        tile_h, tile_w = tile_size
        
    canvas_h = result.rows * tile_h
    canvas_w = result.cols * tile_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    for pl in result.placements:
        if pl.piece_id in pieces:
            piece = rotate_image_90(pieces[pl.piece_id], pl.orientation)
            if piece.shape[:2] != (tile_h, tile_w):
                piece = cv2.resize(piece, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            y0 = pl.row * tile_h
            x0 = pl.col * tile_w
            canvas[y0:y0+tile_h, x0:x0+tile_w] = piece
            
    return canvas






