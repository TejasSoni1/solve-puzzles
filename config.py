"""
Configuration constants for the Jigsaw Puzzle Solver project.

This file serves as a central repository for all hardcoded paths and default settings.
It ensures that file paths are relative to the project directory, making the code
more portable across different environments.
"""
import os

# Default paths
# BASE_DIR is the directory where this config.py file resides (Proj/v3)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

# Default original image (can be overridden by GUI)
# This image is used as the reference for the SIFT solver and for generating pieces
ORIGINAL_PATH = os.path.join(TEST_IMAGES_DIR, "Whole", "william-melek-2.jpg")

# Folder to store generated puzzle pieces
# The image_splitter.py script will output files here, and solvers will read from here
PIECES_FOLDER = os.path.join(TEST_IMAGES_DIR, "pieces")

# Default grid size for splitting the image
# These values determine the difficulty of the puzzle
ROWS = 8
COLS = 12