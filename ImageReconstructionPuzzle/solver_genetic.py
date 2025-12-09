import cv2
import numpy as np
import random
import heapq
import math
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass

import config
from jigsawlver import (
    load_piece_images_from_folder,
    ensure_square_and_same_size,
    rotate_image_90,
    PiecePlacement,
    PuzzleResult
)

# ------------------------------------------------------------
# Algorithm 3: Genetic Algorithm (GAPS)
# Reference: https://github.com/nemanja-m/gaps
# ------------------------------------------------------------

# Constants for priority queue
SHARED_PIECE_PRIORITY = -10
BUDDY_PIECE_PRIORITY = -1

def complementary_orientation(orientation: int) -> int:
    """
    Returns the opposite orientation.
    0 (Top) <-> 2 (Bottom)
    1 (Right) <-> 3 (Left)
    """
    return (orientation + 2) % 4

class ImageAnalysis:
    """
    Static helper class to pre-compute compatibility scores between all pieces.
    This avoids re-calculating edge differences during the genetic evolution.
    """
    dissimilarity_measures: Dict[Tuple[int, int, int, int, int], float] = {}
    best_match_table_rot: Dict[int, Dict[int, Dict[int, List[Tuple[int, int, float]]]]] = {}
    
    @classmethod
    def analyze_image_with_rotation(cls, pieces: Dict[int, np.ndarray]):
        """
        Computes the dissimilarity (edge difference) between every pair of pieces
        in every possible orientation.
        """
        cls.dissimilarity_measures = {}
        # Structure: [pid][orientation][direction] -> List of matches
        cls.best_match_table_rot = {pid: {ori: {d: [] for d in range(4)} for ori in range(4)} for pid in pieces}
        
        # Convert all pieces to LAB color space for better perceptual difference
        pieces_lab = {pid: [cv2.cvtColor(rotate_image_90(img, i), cv2.COLOR_BGR2LAB).astype(np.float32) for i in range(4)] for pid, img in pieces.items()}
        pids = list(pieces.keys())
        
        # Compare every piece against every other piece
        for pid1 in pids:
            for pid2 in pids:
                if pid1 == pid2: continue
                for ori1 in range(4):
                    for ori2 in range(4):
                        img1, img2 = pieces_lab[pid1][ori1], pieces_lab[pid2][ori2]
                        
                        # Calculate difference for all 4 boundaries
                        # Directions: 0:Top, 1:Right, 2:Bottom, 3:Left
                        comparisons = [
                            (0, (img1[0, :, :] - img2[-1, :, :])), # Top of 1 vs Bottom of 2
                            (1, (img1[:, -1, :] - img2[:, 0, :])), # Right of 1 vs Left of 2
                            (2, (img1[-1, :, :] - img2[0, :, :])), # Bottom of 1 vs Top of 2
                            (3, (img1[:, 0, :] - img2[:, -1, :]))  # Left of 1 vs Right of 2
                        ]
                        
                        for direction, diff in comparisons:
                            # Euclidean distance in LAB space
                            score = np.sqrt(np.sum(np.power(diff, 2)))
                            cls.dissimilarity_measures[(pid1, ori1, pid2, ori2, direction)] = score
                            cls.best_match_table_rot[pid1][ori1][direction].append((pid2, ori2, score))

        # Sort matches by score (lowest difference is best)
        for pid in pids:
            for ori in range(4):
                for d in range(4):
                    cls.best_match_table_rot[pid][ori][d].sort(key=lambda x: x[2])

    @classmethod
    def get_best_match(cls, pid, ori, direction):
        """Returns the single best matching piece for a given edge."""
        return cls.best_match_table_rot[pid][ori][direction][0]

class Individual:
    """
    Represents a single solution (puzzle arrangement) in the population.
    """
    def __init__(self, rows: int, cols: int, pieces: List[int], shuffle: bool = True):
        self.rows, self.cols = rows, cols
        self.pieces = pieces[:] 
        # Genome: List of (piece_id, orientation)
        self.genome = [(pid, random.randint(0, 3) if shuffle else 0) for pid in pieces]
        if shuffle: random.shuffle(self.genome)
        self._fitness = None
        self._piece_map = {pid: i for i, (pid, _) in enumerate(self.genome)}

    def piece_at(self, r, c):
        """Get piece at grid coordinates (r, c)."""
        return self.genome[r * self.cols + c] if 0 <= r < self.rows and 0 <= c < self.cols else None
        
    def edge(self, pid, orientation):
        """Get the neighbor of a specific piece in a specific direction."""
        if pid not in self._piece_map: return None
        r, c = divmod(self._piece_map[pid], self.cols)
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][orientation]
        return self.piece_at(r + dr, c + dc)

    @property
    def fitness(self):
        """
        Calculates fitness based on the compatibility of adjacent pieces.
        Higher fitness = better solution (lower dissimilarity score).
        """
        if self._fitness is None:
            score = 0.0
            for r in range(self.rows):
                for c in range(self.cols):
                    idx = r * self.cols + c
                    p1 = self.genome[idx]
                    # Check Right neighbor
                    if c < self.cols - 1:
                        p2 = self.genome[idx + 1]
                        score += ImageAnalysis.dissimilarity_measures.get((p1[0], p1[1], p2[0], p2[1], 1), float('inf'))
                    # Check Bottom neighbor
                    if r < self.rows - 1:
                        p2 = self.genome[idx + self.cols]
                        score += ImageAnalysis.dissimilarity_measures.get((p1[0], p1[1], p2[0], p2[1], 2), float('inf'))
            
            # Invert score so that 0 difference -> infinite fitness
            self._fitness = 1000.0 / score if score > 0 else float('inf')
        return self._fitness

class Crossover:
    """
    Handles the reproduction step. Combines two parent individuals to create a child.
    Uses a 'kernel' growth strategy to preserve good local structures.
    """
    def __init__(self, p1: Individual, p2: Individual, heuristic_prob: float = 0.35):
        self.p1, self.p2 = p1, p2
        self.heuristic_prob = heuristic_prob
        self.rows, self.cols = p1.rows, p1.cols
        self.kernel = {} # pid -> pos
        self.kernel_data = {} # pid -> (r, c, ori)
        self.taken_positions = set()
        self.candidate_pieces = []
        self.min_r = self.max_r = self.min_c = self.max_c = 0

    def run(self):
        """Executes the crossover process."""
        # Start with a random piece
        start_piece = random.choice(self.p1.genome)
        self.put_piece_to_kernel(start_piece, (0, 0))
        
        # Grow the kernel by adding best-fitting neighbors
        while self.candidate_pieces:
            _, (r, c), (pid, ori) = heapq.heappop(self.candidate_pieces)
            if (r, c) in self.taken_positions or not self.is_in_range((r, c)) or pid in self.kernel: continue
            self.put_piece_to_kernel((pid, ori), (r, c))
            
    def put_piece_to_kernel(self, piece, pos):
        """Places a piece in the child's grid and updates boundaries."""
        pid, ori = piece
        self.kernel[pid] = pos
        self.kernel_data[pid] = (pos[0], pos[1], ori)
        self.taken_positions.add(pos)
        self.min_r, self.max_r = min(self.min_r, pos[0]), max(self.max_r, pos[0])
        self.min_c, self.max_c = min(self.min_c, pos[1]), max(self.max_c, pos[1])
        
        # Add potential neighbors to the priority queue
        for direction, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]): # Top, Right, Bottom, Left
            new_pos = (pos[0] + dr, pos[1] + dc)
            if new_pos not in self.taken_positions and self.is_in_range(new_pos):
                self.add_piece_candidate(pid, ori, direction, new_pos)

    def is_in_range(self, pos):
        """Checks if the current kernel size fits within the target grid dimensions."""
        r, c = pos
        return (max(self.max_r, r) - min(self.min_r, r) + 1 <= self.rows) and \
               (max(self.max_c, c) - min(self.min_c, c) + 1 <= self.cols)

    def add_piece_candidate(self, source_pid, source_ori, direction, pos):
        """
        Determines which piece should be placed next to 'source_pid'.
        Prioritizes:
        1. Shared edges (both parents agree).
        2. 'Buddy' pieces (best mutual match).
        3. Best available match from pre-computed table.
        """
        # 1. Shared Piece (Both parents have the same neighbor)
        shared = self.get_shared_piece(source_pid, source_ori, direction)
        if shared and shared[0] not in self.kernel:
            heapq.heappush(self.candidate_pieces, (SHARED_PIECE_PRIORITY, pos, shared))
            return
            
        if random.random() > self.heuristic_prob: return

        # 2. Buddy Piece (Best match is mutual)
        buddy = self.get_buddy_piece(source_pid, source_ori, direction)
        if buddy and buddy[0] not in self.kernel:
            heapq.heappush(self.candidate_pieces, (BUDDY_PIECE_PRIORITY, pos, buddy))
            return
            
        # 3. Best Matches (Fallback to best visual match)
        matches = [m for m in ImageAnalysis.best_match_table_rot[source_pid][source_ori][direction] if m[0] not in self.kernel]
        if not matches: return

        best_score = matches[0][2]
        ratio = matches[0][2] / matches[1][2] if len(matches) > 1 and matches[1][2] > 1e-6 else (1.0 if len(matches) > 1 and matches[1][2] == matches[0][2] else 0.0)
            
        for i, (pid, ori, score) in enumerate(matches[:3]):
            heapq.heappush(self.candidate_pieces, (ratio + (i * 10.0), pos, (pid, ori)))

    def get_shared_piece(self, pid, ori, direction):
        n1, n2 = self.p1.edge(pid, direction), self.p2.edge(pid, direction)
        return n1 if n1 and n2 and n1 == n2 else None

    def get_buddy_piece(self, pid, ori, direction):
        best = ImageAnalysis.get_best_match(pid, ori, direction)
        if best:
            back = ImageAnalysis.get_best_match(best[0], best[1], complementary_orientation(direction))
            if back and back[0] == pid: return (best[0], best[1])
        return None

    def child(self) -> Individual:
        """Constructs the final child individual from the kernel."""
        child_genome = [None] * (self.rows * self.cols)
        for pid, (r, c, ori) in self.kernel_data.items():
            child_genome[(r - self.min_r) * self.cols + (c - self.min_c)] = (pid, ori)
            
        # Fill any remaining gaps with unused pieces
        unused = [p for p in self.p1.pieces if p not in self.kernel]
        random.shuffle(unused)
        
        for i in range(len(child_genome)):
            if child_genome[i] is None: child_genome[i] = (unused.pop(), random.randint(0, 3))
                    
        ind = Individual(self.rows, self.cols, self.p1.pieces, shuffle=False)
        ind.genome = child_genome
        return ind

class GeneticSolver:
    """
    Main class for the Genetic Algorithm solver.
    Manages population, evolution loop, and visualization.
    """
    def __init__(self, pieces: Dict[int, np.ndarray], rows: int, cols: int, 
                 generations: int = 20, population_size: int = 100, heuristic_rate: float = 0.3):
        self.pieces = ensure_square_and_same_size(pieces)
        self.rows = rows
        self.cols = cols
        self.generations = generations
        self.population_size = population_size
        self.heuristic_rate = heuristic_rate
        self.piece_ids = list(self.pieces.keys())
        
        # Precompute all piece compatibilities once
        ImageAnalysis.analyze_image_with_rotation(self.pieces)

    def create_initial_population(self):
        return [Individual(self.rows, self.cols, self.piece_ids, shuffle=True) 
                for _ in range(self.population_size)]

    def selection(self, population):
        """Tournament selection: Pick k random, return the best."""
        k = 5
        selected = random.sample(population, k)
        selected.sort(key=lambda x: x.fitness, reverse=True)
        return selected[0]

    def mutate(self, ind: Individual):
        """Applies random mutations (swap pieces or rotate piece)."""
        if random.random() < 0.1:
            # Swap two pieces
            idx1, idx2 = random.sample(range(len(ind.genome)), 2)
            ind.genome[idx1], ind.genome[idx2] = ind.genome[idx2], ind.genome[idx1]
            ind._fitness = None
            ind._piece_map = {pid: i for i, (pid, _) in enumerate(ind.genome)}
            
        if random.random() < 0.1:
            # Rotate a piece
            idx = random.randint(0, len(ind.genome) - 1)
            pid, ori = ind.genome[idx]
            ind.genome[idx] = (pid, (ori + 1) % 4)
            ind._fitness = None

    def visualize_individual(self, individual: Individual) -> np.ndarray:
        """Creates an image representation of the individual's genome."""
        if not self.pieces:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        first_piece = next(iter(self.pieces.values()))
        tile_h, tile_w = first_piece.shape[:2]
        
        canvas_h = self.rows * tile_h
        canvas_w = self.cols * tile_w
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        for r in range(self.rows):
            for c in range(self.cols):
                p_info = individual.piece_at(r, c)
                if p_info:
                    pid, ori = p_info
                    if pid in self.pieces:
                        piece = rotate_image_90(self.pieces[pid], ori)
                        # Ensure piece fits
                        if piece.shape[:2] != (tile_h, tile_w):
                            piece = cv2.resize(piece, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                        
                        y0 = r * tile_h
                        x0 = c * tile_w
                        canvas[y0:y0+tile_h, x0:x0+tile_w] = piece
        return canvas

    def solve(self, on_progress=None):
        """Main evolution loop."""
        population = self.create_initial_population()
        best_ind = max(population, key=lambda x: x.fitness)
        
        for gen in range(self.generations):
            new_pop = []
            new_pop.append(best_ind) # Elitism: Keep the best one
            
            while len(new_pop) < self.population_size:
                p1 = self.selection(population)
                p2 = self.selection(population)
                
                # Use the fixed crossover
                crossover = Crossover(p1, p2, heuristic_prob=self.heuristic_rate)
                crossover.run()
                child = crossover.child()
                
                self.mutate(child)
                new_pop.append(child)
                
            population = new_pop
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_ind.fitness:
                best_ind = current_best
            
            print(f"Gen {gen}: Best Fitness {best_ind.fitness:.4f}")
            if on_progress:
                on_progress(gen, best_ind)
            
        placements = []
        for r in range(self.rows):
            for c in range(self.cols):
                pid, ori = best_ind.piece_at(r, c)
                placements.append(PiecePlacement(pid, ori, r, c))
                
        return PuzzleResult(self.rows, self.cols, placements)

def solve_genetic(pieces_folder: str, rows: int, cols: int, generations: int = 20, population: int = 100, heuristic_rate: float = 0.35) -> PuzzleResult:
    pieces = load_piece_images_from_folder(pieces_folder)
    solver = GeneticSolver(pieces, rows, cols, generations, population, heuristic_rate)
    return solver.solve()

