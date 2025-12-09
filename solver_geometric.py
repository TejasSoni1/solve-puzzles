import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional, Set

import config
from jigsawlver import (
    load_piece_images_from_folder,
    rotate_image_90,
    ensure_square_and_same_size,
    PiecePlacement,
    PuzzleResult
)

# ------------------------------------------------------------
# Algorithm 2: Without Original Image (Sobel + Greedy)
# ------------------------------------------------------------

class JigsawSolverNoOriginal:
    """
    Implements the 'Algorithm without the Original Image' (Section 2.2).
    
    Strategy:
    1. Edge Compatibility: Use Sobel filter to detect edges and compare them.
    2. Best Piece (BP): Find the piece with the most distinct/matchable edges.
    3. Initial 3x3: Build a core cluster around the Best Piece.
    4. Expansion: Greedily add rings of pieces around the core until full.
    """

    # Sobel filter kernel for edge detection
    SOBEL_KERNEL = np.array([
        [-1, 1],
        [-2, 2],
        [-1, 1]
    ], dtype=np.float32)

    def __init__(self,
                 pieces: Dict[int, np.ndarray],
                 rows: Optional[int] = None,
                 cols: Optional[int] = None,
                 threshold_ratio: float = 0.5):

        self.rows = rows
        self.cols = cols
        self.threshold_ratio = threshold_ratio

        # Normalize piece sizes to ensure consistent edge comparison
        pieces = ensure_square_and_same_size(pieces)
        self.piece_size = next(iter(pieces.values())).shape[0]

        # Store grayscale versions for processing
        self.pieces_gray: Dict[int, np.ndarray] = {}
        for pid, img in pieces.items():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
            self.pieces_gray[pid] = gray.astype(np.float32)

        self.num_pieces = len(self.pieces_gray)

        # Grid representation: (row, col) -> (piece_id, orientation)
        self.grid: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.used_pieces: Set[int] = set()

        # Bounds of current grid (inclusive)
        self.min_r = 0
        self.max_r = 0
        self.min_c = 0
        self.max_c = 0

    # -------------------- Edge & similarity utilities --------------------

    def _get_oriented_piece(self, piece_id: int, orientation: int) -> np.ndarray:
        """Return grayscale oriented piece."""
        img = self.pieces_gray[piece_id]
        return rotate_image_90(img, orientation)

    def _get_edge_1d(self, img: np.ndarray, side: int) -> np.ndarray:
        """
        Extracts the pixel values of a specific edge.
        side: 0=top, 1=right, 2=bottom, 3=left.
        """
        if side == 0: return img[0, :].astype(np.float32)
        if side == 1: return img[:, -1].astype(np.float32)
        if side == 2: return img[-1, :].astype(np.float32)
        if side == 3: return img[:, 0].astype(np.float32)
        raise ValueError("side must be 0..3")

    def _sobel_similarity(self, edge_left: np.ndarray, edge_right: np.ndarray) -> float:
        """
        Calculates similarity between two edges using Sobel derivative.
        Lower derivative difference -> Higher similarity.
        Formula: 1 / (1 + || H * (el | er) ||)
        """
        # Concatenate edges to form a boundary
        E = np.stack([edge_left - np.mean(edge_left), edge_right - np.mean(edge_right)], axis=1)
        # Apply Sobel filter to detect abrupt changes across the boundary
        norm = float(np.linalg.norm(cv2.filter2D(E, ddepth=cv2.CV_32F, kernel=self.SOBEL_KERNEL)))
        return 1.0 / (1.0 + norm)

    def edge_similarity_for_adjacent(self, pid_a: int, ori_a: int, side_a: int, pid_b: int, ori_b: int, side_b: int) -> float:
        """Helper to get similarity between specific sides of two oriented pieces."""
        return self._sobel_similarity(self._get_edge_1d(self._get_oriented_piece(pid_a, ori_a), side_a),
                                      self._get_edge_1d(self._get_oriented_piece(pid_b, ori_b), side_b))

    # -------------------- Best Piece (BP) search --------------------

    def find_best_piece(self) -> int:
        """
        Finds the 'Best Piece' to start the puzzle.
        The BP is the piece that has the highest minimum similarity score across all its 4 sides.
        """
        best_scores = {}
        for pid in self.pieces_gray.keys():
            # Check all 4 sides: 1(Right), 3(Left), 2(Bottom), 0(Top) against their neighbors
            scores = [self._best_side_similarity(pid, 0, side, (side + 2) % 4) for side in [1, 3, 2, 0]]
            best_scores[pid] = min(scores)
        return max(best_scores.items(), key=lambda kv: kv[1])[0]

    def _best_side_similarity(self, pid_center: int, ori_center: int, center_side: int, neighbor_side: int) -> float:
        """Finds the best possible match score for a specific side of a piece."""
        best = 0.0
        for pid_other in self.pieces_gray.keys():
            if pid_other == pid_center: continue
            for ori_other in range(4):
                s = self.edge_similarity_for_adjacent(pid_center, ori_center, center_side, pid_other, ori_other, neighbor_side)
                if s > best: best = s
        return best

    # -------------------- Initial 3x3 combination (cross + corners) --------------------

    def build_initial_3x3(self, bp: int):
        """
        Builds a 3x3 grid starting from the Best Piece (BP) at the center.
        1. Place BP.
        2. Place 4 direct neighbors (Cross).
        3. Place 4 corner neighbors.
        """
        self.grid.clear()
        self.used_pieces = {bp}
        self.grid[(0, 0)] = (bp, 0)
        self.min_r = self.max_r = self.min_c = self.max_c = 0

        # 1. Cross neighbors: (dr, dc, center_side, neigh_side)
        neighbors_config = [(-1, 0, 0, 2), (1, 0, 2, 0), (0, -1, 3, 1), (0, 1, 1, 3)]
        
        for dr, dc, c_side, n_side in neighbors_config:
            best = (-1, -1, -1.0)
            for pid in self.pieces_gray.keys():
                if pid in self.used_pieces: continue
                for ori in range(4):
                    s = self.edge_similarity_for_adjacent(bp, 0, c_side, pid, ori, n_side)
                    if s > best[2]: best = (pid, ori, s)
            
            if best[0] != -1:
                pos = (dr, dc)
                self.grid[pos] = (best[0], best[1])
                self.used_pieces.add(best[0])
                self.min_r, self.max_r = min(self.min_r, dr), max(self.max_r, dr)
                self.min_c, self.max_c = min(self.min_c, dc), max(self.max_c, dc)

        # 2. Corners: Fill gaps based on two existing neighbors
        # Format: (pos, n1_pos, n1_side, n2_pos, n2_side)
        corners = [
            ((-1, -1), (-1, 0), 1, (0, -1), 2), # TL
            ((-1, 1), (-1, 0), 3, (0, 1), 2),   # TR
            ((1, -1), (1, 0), 1, (0, -1), 0),   # BL
            ((1, 1), (1, 0), 3, (0, 1), 0)      # BR
        ]

        for pos, n1_pos, side1, n2_pos, side2 in corners:
            if n1_pos not in self.grid or n2_pos not in self.grid: continue
            
            best = (-1, -1, -1.0)
            pid1, ori1 = self.grid[n1_pos]
            pid2, ori2 = self.grid[n2_pos]
            
            for pid in self.pieces_gray.keys():
                if pid in self.used_pieces: continue
                for ori in range(4):
                    # Average score with both neighbors
                    s1 = self.edge_similarity_for_adjacent(pid, ori, side1, pid1, ori1, (side1 + 2) % 4)
                    s2 = self.edge_similarity_for_adjacent(pid, ori, side2, pid2, ori2, (side2 + 2) % 4)
                    if (s1 + s2) / 2.0 > best[2]: best = (pid, ori, (s1 + s2) / 2.0)
            
            if best[0] != -1:
                self.grid[pos] = (best[0], best[1])
                self.used_pieces.add(best[0])

        self.min_r, self.max_r = min(self.min_r, -1), max(self.max_r, 1)
        self.min_c, self.max_c = min(self.min_c, -1), max(self.max_c, 1)

    @staticmethod
    def _opposite_side(side: int) -> int:
        return (side + 2) % 4

    # -------------------- Square expansion (initial combination) --------------------

    def expand_square_until_all_pieces(self):
        """
        Expands the grid by adding 'rings' of pieces.
        Decides which direction (Top, Bottom, Left, Right) is best to expand next.
        """
        while len(self.used_pieces) < self.num_pieces:
            scores = self._compute_side_scores()
            best_dir = max(scores.items(), key=lambda kv: kv[1])[0]
            self._add_ring(best_dir)

    def _compute_side_scores(self) -> Dict[str, float]:
        """Computes the average similarity score for adding a row/col to each side."""
        scores = {}
        # Config: (name, fixed_coord, is_row_fixed, start, end, new_side, existing_side)
        configs = [
            ("top", self.min_r, True, self.min_c, self.max_c, 2, 0),
            ("bottom", self.max_r, True, self.min_c, self.max_c, 0, 2),
            ("left", self.min_c, False, self.min_r, self.max_r, 1, 3),
            ("right", self.max_c, False, self.min_r, self.max_r, 3, 1)
        ]
        
        for name, fixed, is_row, start, end, new_s, exist_s in configs:
            sim_list = []
            for i in range(start, end + 1):
                pos = (fixed, i) if is_row else (i, fixed)
                if pos not in self.grid: continue
                pid, ori = self.grid[pos]
                best = 0.0
                for pid2 in self.pieces_gray.keys():
                    if pid2 in self.used_pieces: continue
                    for ori2 in range(4):
                        s = self.edge_similarity_for_adjacent(pid2, ori2, new_s, pid, ori, exist_s)
                        if s > best: best = s
                if best > 0: sim_list.append(best)
            scores[name] = float(np.mean(sim_list)) if sim_list else 0.0
        return scores

    def _add_ring(self, direction: str):
        """Adds a row or column of pieces in the specified direction."""
        if direction == "top":
            self.min_r -= 1
            for c in range(self.min_c, self.max_c + 1): self._assign_best_piece_to_position(self.min_r, c, (self.min_r + 1, c), 2, 0)
        elif direction == "bottom":
            self.max_r += 1
            for c in range(self.min_c, self.max_c + 1): self._assign_best_piece_to_position(self.max_r, c, (self.max_r - 1, c), 0, 2)
        elif direction == "left":
            self.min_c -= 1
            for r in range(self.min_r, self.max_r + 1): self._assign_best_piece_to_position(r, self.min_c, (r, self.min_c + 1), 1, 3)
        elif direction == "right":
            self.max_c += 1
            for r in range(self.min_r, self.max_r + 1): self._assign_best_piece_to_position(r, self.max_c, (r, self.max_c - 1), 3, 1)

    def _assign_best_piece_to_position(self, r: int, c: int, neighbor_pos: Tuple[int, int], new_side: int, neighbor_side: int):
        """Finds and places the best available piece for a specific grid position."""
        if neighbor_pos not in self.grid: return
        pid_neigh, ori_neigh = self.grid[neighbor_pos]
        best = (-1, -1, -1.0)
        for pid in self.pieces_gray.keys():
            if pid in self.used_pieces: continue
            for ori in range(4):
                s = self.edge_similarity_for_adjacent(pid, ori, new_side, pid_neigh, ori_neigh, neighbor_side)
                if s > best[2]: best = (pid, ori, s)
        
        if best[0] != -1 and best[2] > 0:
            self.grid[(r, c)] = (best[0], best[1])
            self.used_pieces.add(best[0])

    # -------------------- Main track (threshold filtering) --------------------

    def build_main_track(self):
        """
        Filters the grid to keep only the 'Main Track' - pieces with high confidence connections.
        Removes pieces that fall below a similarity threshold.
        """
        # Compute all adjacency similarities
        adj_scores = {}  # (r,c) -> list of similarities with neighbors
        all_sims = []

        for (r, c), (pid, ori) in self.grid.items():
            sims_here = []
            for dr, dc, side_self, side_nb in [(-1, 0, 0, 2), (1, 0, 2, 0),
                                               (0, -1, 3, 1), (0, 1, 1, 3)]:
                nb_pos = (r + dr, c + dc)
                if nb_pos not in self.grid:
                    continue
                pid_nb, ori_nb = self.grid[nb_pos]
                s = self.edge_similarity_for_adjacent(pid, ori, side_self,
                                                      pid_nb, ori_nb, side_nb)
                sims_here.append(s)
                all_sims.append(s)
            if sims_here:
                adj_scores[(r, c)] = sims_here

        if not all_sims:
            return

        global_mean = float(np.mean(all_sims))
        threshold = self.threshold_ratio * global_mean

        # Build new grid with only strong pieces
        new_grid = {}
        new_used = set()
        for (r, c), sims in adj_scores.items():
            if min(sims) >= threshold:
                pid, ori = self.grid[(r, c)]
                new_grid[(r, c)] = (pid, ori)
                new_used.add(pid)

        self.grid = new_grid
        self.used_pieces = new_used

        if self.grid:
            rs = [rc[0] for rc in self.grid.keys()]
            cs = [rc[1] for rc in self.grid.keys()]
            self.min_r, self.max_r = min(rs), max(rs)
            self.min_c, self.max_c = min(cs), max(cs)

    # -------------------- Second combination (fill around main track) --------------------

    def second_combination(self):
        """
        Greedy filling of remaining pieces around the main track.
        
        This method iteratively finds the best piece to place in any open spot
        adjacent to the current puzzle assembly. It scores potential placements
        based on edge compatibility with neighbors.
        """
        remaining = set(self.pieces_gray.keys()) - self.used_pieces

        # Helper to find empty spots next to existing pieces
        def get_frontier_positions():
            frontier = set()
            for (r, c) in self.grid.keys():
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    pos = (r + dr, c + dc)
                    if pos not in self.grid:
                        frontier.add(pos)
            return list(frontier)

        while remaining:
            frontier = get_frontier_positions()
            if not frontier: break

            best_move = (-1, -1, (-1, -1), -1.0) # (pid, ori, pos, score)

            for pos in frontier:
                r, c = pos
                # Check neighbors to ensure we have something to compare against
                neighbors = []
                for dr, dc, s_self, s_nb in [(-1, 0, 0, 2), (1, 0, 2, 0), (0, -1, 3, 1), (0, 1, 1, 3)]:
                    nb_pos = (r + dr, c + dc)
                    if nb_pos in self.grid:
                        neighbors.append((self.grid[nb_pos], s_self, s_nb))
                
                if not neighbors: continue

                # Try every remaining piece in every orientation at this position
                for pid in remaining:
                    for ori in range(4):
                        score_sum = 0
                        count = 0
                        for (n_pid, n_ori), s_self, s_nb in neighbors:
                            score_sum += self.edge_similarity_for_adjacent(pid, ori, s_self, n_pid, n_ori, s_nb)
                            count += 1
                        
                        avg_score = score_sum / count if count > 0 else 0
                        if avg_score > best_move[3]:
                            best_move = (pid, ori, pos, avg_score)

            # Apply the best move found in this iteration
            if best_move[0] != -1:
                pid, ori, pos, score = best_move
                self.grid[pos] = (pid, ori)
                self.used_pieces.add(pid)
                remaining.remove(pid)
                self.min_r, self.max_r = min(self.min_r, pos[0]), max(self.max_r, pos[0])
                self.min_c, self.max_c = min(self.min_c, pos[1]), max(self.max_c, pos[1])
            else:
                break

    def _placement_score(self, pid: int, ori: int, r: int, c: int) -> float:
        """
        Calculate the average similarity score for placing a piece at (r, c).
        Used by third_combination.
        """
        sims = []
        for (dr, dc, side_self, side_nb) in [(-1, 0, 0, 2), (1, 0, 2, 0), (0, -1, 3, 1), (0, 1, 1, 3)]:
            nb_pos = (r + dr, c + dc)
            if nb_pos in self.grid:
                pid_nb, ori_nb = self.grid[nb_pos]
                s = self.edge_similarity_for_adjacent(pid, ori, side_self, pid_nb, ori_nb, side_nb)
                sims.append(s)
        return float(np.mean(sims)) if sims else 0.0

    def third_combination(self):
        """
        Aspect Ratio Correction.
        
        If the target dimensions (rows x cols) are known, this step attempts to
        crop the current assembly to the best matching window and re-place
        any excluded pieces into the empty spots within that window.
        """
        if self.rows is None or self.cols is None:
            return

        if not self.grid:
            return
            
        # Calculate current bounds
        rs = [rc[0] for rc in self.grid.keys()]
        cs = [rc[1] for rc in self.grid.keys()]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        H = max_r - min_r + 1
        W = max_c - min_c + 1

        R, C = self.rows, self.cols
        if H == R and W == C:
            return # Already correct shape

        # Find the R x C window with the most pieces
        best_window = None
        best_count = -1
        for r0 in range(min_r, max_r - R + 2):
            for c0 in range(min_c, max_c - C + 2):
                count = 0
                for (r, c) in self.grid.keys():
                    if r0 <= r < r0 + R and c0 <= c < c0 + C:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_window = (r0, c0)

        if best_window is None:
            return

        r0, c0 = best_window

        # Keep pieces inside the window, remove others
        new_grid = {}
        fixed_pieces = set()
        for (r, c), val in self.grid.items():
            if r0 <= r < r0 + R and c0 <= c < c0 + C:
                new_grid[(r, c)] = val
                fixed_pieces.add(val[0])

        removed_pieces = set(pid for pid, _ in self.grid.values() if pid not in fixed_pieces)

        self.grid = new_grid
        self.used_pieces = fixed_pieces
        self.min_r, self.max_r = r0, r0 + R - 1
        self.min_c, self.max_c = c0, c0 + C - 1

        # Re-fill empty spots in the window with removed/unused pieces
        remaining = removed_pieces | (set(self.pieces_gray.keys()) - self.used_pieces)
        empty_cells = [(r, c) for r in range(self.min_r, self.max_r + 1)
                       for c in range(self.min_c, self.max_c + 1)
                       if (r, c) not in self.grid]

        while remaining and empty_cells:
            best_global = (None, None, None, -1.0)
            for pos in empty_cells:
                r, c = pos
                for pid in list(remaining):
                    for ori in range(4):
                        score = self._placement_score(pid, ori, r, c)
                        if score > best_global[3]:
                            best_global = (pos, pid, ori, score)

            pos_best, pid_best, ori_best, score_best = best_global
            if pos_best is None or score_best <= 0:
                break

            self.grid[pos_best] = (pid_best, ori_best)
            self.used_pieces.add(pid_best)
            remaining.remove(pid_best)
            empty_cells.remove(pos_best)

def solve_without_original(pieces_folder: str, rows: int, cols: int) -> PuzzleResult:
    """
    Main entry point for the geometric solver (No Original Image).
    
    Orchestrates the solving process:
    1. Load pieces
    2. Find a starting piece
    3. Build a core
    4. Expand
    5. Fill gaps
    6. Correct aspect ratio
    """
    pieces = load_piece_images_from_folder(pieces_folder)
    solver = JigsawSolverNoOriginal(pieces, rows, cols)
    
    # Step 1: Find best starting piece (most "corner-like" or distinct)
    bp = solver.find_best_piece()
    
    # Step 2: Build initial 3x3 core around the start piece
    solver.build_initial_3x3(bp)
    
    # Step 3: Expand outwards in a square fashion
    solver.expand_square_until_all_pieces()
    
    # Step 4: Build the main track (strongest edge matches)
    solver.build_main_track()
    
    # Step 5: Fill in remaining gaps greedily
    solver.second_combination()
    
    # Step 6: Enforce aspect ratio constraints if possible
    solver.third_combination()

    # Convert internal grid representation to PuzzleResult
    placements = []
    for (r, c), (pid, ori) in solver.grid.items():
        # Normalize coordinates to start at (0,0)
        placements.append(PiecePlacement(r - solver.min_r, c - solver.min_c, pid, ori))
        
    return PuzzleResult(
        rows=solver.max_r - solver.min_r + 1,
        cols=solver.max_c - solver.min_c + 1,
        placements=placements
    )