
import os
import cv2
import numpy as np
import random

import config
from jigsawlver import rotate_image_90, load_piece_images_from_folder, load_image

def split_and_shuffle_image(image_path=None, output_folder=None, rows=None, cols=None, progress_callback=None):
    """
    Splits a source image into a grid of puzzle pieces, shuffles them, and applies random rotations.
    
    This function simulates the creation of a jigsaw puzzle by:
    1. Loading the source image.
    2. Dividing it into a grid of (rows x cols) tiles.
    3. Saving each tile as a separate image file.
    4. Reloading the tiles, applying random 90-degree rotations, and saving them with shuffled names.
    5. Cleaning up the temporary ordered files.
    
    Args:
        image_path (str, optional): Path to the source image. Defaults to config.ORIGINAL_PATH.
        output_folder (str, optional): Directory to save the generated pieces. Defaults to config.PIECES_FOLDER.
        rows (int, optional): Number of rows in the grid. Defaults to config.ROWS.
        cols (int, optional): Number of columns in the grid. Defaults to config.COLS.
        progress_callback (callable, optional): Function to call with (current_piece_index, total_pieces) for UI updates.
    """
    image_path = image_path or config.ORIGINAL_PATH
    output_folder = output_folder or config.PIECES_FOLDER
    rows = rows or config.ROWS
    cols = cols or config.COLS

    os.makedirs(output_folder, exist_ok=True)

    # --- 1) Load and cut image into tiles ---
    img = load_image(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]
    tile_h = h // rows
    tile_w = w // cols

    # Clear the folder of any existing images to avoid mixing old and new puzzles
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            os.remove(file_path)

    piece_paths = []
    pid = 0
    total_pieces = rows * cols
    
    # Iterate through the grid and extract each tile
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            tile = img[y0:y1, x0:x1]

            # Save temporarily with ordered names
            out_path = os.path.join(output_folder, f"piece_{pid:04d}.png")
            cv2.imwrite(out_path, tile)
            piece_paths.append(out_path)
            pid += 1
            
            if progress_callback:
                progress_callback(pid, total_pieces)

    print(f"Saved {pid} tiles to {output_folder}")

    # --- 2) Randomly rotate and shuffle filenames to simulate a real puzzle ---
    pieces = load_piece_images_from_folder(output_folder)
    for pid, im in pieces.items():
        # Random rotation (0, 90, 180, or 270 degrees)
        k = random.randint(0, 3)
        rot = rotate_image_90(im, k)
        
        # Square Rescaling: Ensure all pieces are the same size after rotation (important for non-square tiles)
        if rot.shape[:2] != (tile_h, tile_w):
            rot = cv2.resize(rot, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            
        # Save with a new "shuffled" prefix. 
        cv2.imwrite(os.path.join(output_folder, f"shuffled_{pid:04d}.png"), rot)

    # --- 3) Cleanup ---
    # Delete the original ordered tiles so only the rotated/shuffled versions remain
    for p in piece_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    print("Shuffled pieces saved.")

if __name__ == "__main__":
    split_and_shuffle_image()
