import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import project modules
import config
from image_splitter import split_and_shuffle_image
from jigsawlver import load_image, load_piece_images_from_folder, visualize_puzzle_result
from solver_sift import JigsawSolverWithOriginal
from solver_geometric import solve_without_original
from solver_genetic import GeneticSolver

class JigsawGUI:
    """
    Main GUI Application for the Jigsaw Puzzle Solver.
    Handles user interaction, image selection, and visualization of the solving process.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Jigsaw Puzzle Solver")
        self.root.geometry("1200x800") # Increased width for plot
        self.root.configure(bg="#f0f0f0")

        # Variables to store user inputs
        self.image_path = tk.StringVar()
        self.rows = tk.IntVar(value=8)
        self.cols = tk.IntVar(value=12)
        self.pop_var = tk.IntVar(value=600) # Population size for GA
        self.gen_var = tk.IntVar(value=100) # Generations for GA
        self.method = tk.StringVar(value="with_original") # Default method

        # Initialize UI Elements
        self.create_widgets()

    def create_widgets(self):
        """Creates and packs all GUI widgets."""
        # Title
        tk.Label(self.root, text="Jigsaw Puzzle Solver", font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#333").pack(pady=20)

        # Helper function to create consistent input rows
        def create_row(parent, label_text, var, width=10, btn_cmd=None):
            frame = tk.Frame(parent, bg="#f0f0f0")
            frame.pack(pady=5)
            tk.Label(frame, text=label_text, font=("Arial", 12), bg="#f0f0f0").pack(side="left")
            tk.Entry(frame, textvariable=var, width=width).pack(side="left", padx=10)
            if btn_cmd: tk.Button(frame, text="Browse", command=btn_cmd, bg="#4CAF50", fg="white").pack(side="left")
            return frame

        # Image Selection Row
        create_row(self.root, "Select Image:", self.image_path, 50, self.browse_image)
        
        # Grid Dimensions Row
        frame_dims = tk.Frame(self.root, bg="#f0f0f0")
        frame_dims.pack(pady=5)
        for txt, var in [("Rows:", self.rows), ("Cols:", self.cols)]:
            tk.Label(frame_dims, text=txt, font=("Arial", 12), bg="#f0f0f0").pack(side="left")
            tk.Entry(frame_dims, textvariable=var, width=10).pack(side="left", padx=10)

        # Genetic Algorithm Settings Row
        frame_ga = tk.Frame(self.root, bg="#f0f0f0")
        frame_ga.pack(pady=5)
        for txt, var in [("GA Pop:", self.pop_var), ("GA Gens:", self.gen_var)]:
            tk.Label(frame_ga, text=txt, font=("Arial", 10), bg="#f0f0f0").pack(side="left")
            tk.Entry(frame_ga, textvariable=var, width=8).pack(side="left", padx=5)

        # Method Selection Radio Buttons
        frame_method = tk.Frame(self.root, bg="#f0f0f0")
        frame_method.pack(pady=10)
        tk.Label(frame_method, text="Solving Method:", font=("Arial", 12), bg="#f0f0f0").pack(anchor="w")
        for text, val in [("With Original Image (SIFT + RANSAC)", "with_original"), 
                          ("Without Original Image (Sobel + Greedy)", "without_original"), 
                          ("Genetic Algorithm", "genetic")]:
            tk.Radiobutton(frame_method, text=text, variable=self.method, value=val, bg="#f0f0f0").pack(anchor="w")

        # Start Button
        tk.Button(self.root, text="Start Solving", command=self.start_process, bg="#2196F3", fg="white", font=("Arial", 14, "bold"), padx=20, pady=10).pack(pady=10)

        # Visualization Frame (Split View: Image on Left, Plot on Right)
        self.viz_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.viz_frame.pack(pady=5, fill="both", expand=True)
        
        # Image Display (Left)
        self.result_label = tk.Label(self.viz_frame, bg="#f0f0f0")
        self.result_label.pack(side="left", padx=10, expand=True)
        
        # Plot Display (Right) - Matplotlib integration
        self.plot_frame = tk.Frame(self.viz_frame, bg="#f0f0f0")
        self.plot_frame.pack(side="right", padx=10, fill="both", expand=True)
        
        # Initialize Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.ax.set_title("Fitness over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Progress Bar and Log Output
        self.progress_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.progress_frame.pack(pady=5, fill="x")
        self.progress_label = tk.Label(self.progress_frame, text="", font=("Arial", 10), bg="#f0f0f0")
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5)
        
        self.output_text = tk.Text(self.progress_frame, height=5, width=80, state="disabled")
        self.output_text.pack(pady=5)

    def browse_image(self):
        """Opens file dialog to select source image."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image_path.set(file_path)

    def start_process(self):
        """Validates inputs and starts the solving process in a separate thread."""
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image.")
            return
        if self.rows.get() <= 0 or self.cols.get() <= 0:
            messagebox.showerror("Error", "Rows and Cols must be positive integers.")
            return

        # Update global config with user inputs
        config.ORIGINAL_PATH = self.image_path.get()
        config.ROWS = self.rows.get()
        config.COLS = self.cols.get()

        # Reset UI state
        self.progress_bar['value'] = 0
        self.progress_label.config(text="")
        self.output_text.config(state="normal")
        
        # Capture GA settings
        self.ga_pop = self.pop_var.get()
        self.ga_gen = self.gen_var.get()
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state="disabled")
        self.result_label.config(image="")
        
        # Reset Plot Data
        self.fitness_history = []
        self.gen_history = []
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        # Run in thread to avoid freezing UI during heavy computation
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        """Main execution logic: Split image -> Solve -> Visualize."""
        try:
            self.update_progress("Splitting image...", 0)
            self.log_output("Starting image splitting...")

            # Step 1: Split the source image into puzzle pieces
            split_and_shuffle_image(
                image_path=config.ORIGINAL_PATH,
                output_folder=config.PIECES_FOLDER,
                rows=config.ROWS,
                cols=config.COLS,
                progress_callback=lambda c, t: self.update_progress(f"Splitting: {c}/{t}", (c / t) * 50)
            )

            self.log_output(f"Saved tiles to {config.PIECES_FOLDER}")
            self.log_output("Shuffled pieces.")

            self.update_progress("Solving puzzle...", 50)

            # Step 2: Solve using selected method
            if self.method.get() == "with_original":
                self.log_output("Solving with original image...")
                
                orig = load_image(config.ORIGINAL_PATH)
                pieces = load_piece_images_from_folder(config.PIECES_FOLDER)
                solver = JigsawSolverWithOriginal(orig)
                
                # Solve and get reconstructed image
                canvas, transforms = solver.solve_and_reconstruct(pieces)
                
                for pid, (M, angle_deg, (cx, cy)) in transforms.items():
                    self.log_output(f"Piece {pid}: angle={angle_deg:.1f}°, center=({cx:.1f}, {cy:.1f})")

                solved_path = os.path.join(os.path.dirname(config.PIECES_FOLDER), "solved_with_original.png")
                cv2.imwrite(solved_path, canvas)
                self.display_image(solved_path)

            elif self.method.get() == "genetic":
                self.log_output("Solving with Genetic Algorithm...")
                pieces = load_piece_images_from_folder(config.PIECES_FOLDER)
                
                # Initialize Genetic Solver
                solver = GeneticSolver(pieces, rows=config.ROWS, cols=config.COLS, population_size=self.ga_pop, generations=self.ga_gen)
                
                # Callback for live updates during GA evolution
                def on_gen_update(gen, best_ind):
                    try:
                        canvas = solver.visualize_individual(best_ind)
                        
                        # Convert to RGB for display (OpenCV uses BGR)
                        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                        
                        # Schedule GUI update on main thread
                        self.root.after(0, lambda: self.update_live_view(canvas_rgb, gen, best_ind.fitness))
                    except Exception as e:
                        print(f"Error in visualization: {e}")

                result = solver.solve(on_progress=on_gen_update)
                
                # Final Visualization
                canvas = visualize_puzzle_result(result, pieces)

                solved_path = os.path.join(os.path.dirname(config.PIECES_FOLDER), "solved_genetic.png")
                cv2.imwrite(solved_path, canvas)
                self.display_image(solved_path)

            else:
                self.log_output("Solving without original image...")
                result = solve_without_original(config.PIECES_FOLDER, rows=config.ROWS, cols=config.COLS)
                self.log_output(f"Rows: {result.rows}, Cols: {result.cols}")
                for pl in result.placements[:10]:
                    self.log_output(f"Placement: {pl}")

                # Visualize Result
                pieces = load_piece_images_from_folder(config.PIECES_FOLDER)
                canvas = visualize_puzzle_result(result, pieces)

                solved_path = os.path.join(os.path.dirname(config.PIECES_FOLDER), "solved_without_original.png")
                cv2.imwrite(solved_path, canvas)
                self.display_image(solved_path)

            self.update_progress("Done!", 100)
            self.log_output("Puzzle solved successfully!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log_output(f"Error: {str(e)}")

    def update_live_view(self, canvas_rgb, gen, fitness):
        """Updates the GUI with the current best solution and fitness plot."""
        try:
            img = Image.fromarray(canvas_rgb)
            # Resize to fit the window while maintaining aspect ratio
            display_w, display_h = 600, 450
            img.thumbnail((display_w, display_h), Image.Resampling.NEAREST)
            
            img_tk = ImageTk.PhotoImage(img)
            self.result_label.config(image=img_tk)
            self.result_label.image = img_tk
            
            # Update Fitness Plot
            self.gen_history.append(gen)
            self.fitness_history.append(fitness)
            self.line.set_data(self.gen_history, self.fitness_history)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            
            progress_pct = (gen / self.ga_gen) * 50 + 50
            self.update_progress(f"Generation {gen}: Fitness {fitness:.2f}", progress_pct)
        except Exception as e:
            print(f"Error updating GUI: {e}")

    def update_progress(self, text, value):
        """Updates the progress bar and label."""
        self.progress_label.config(text=text)
        self.progress_bar['value'] = value
        self.root.update_idletasks()

    def log_output(self, text):
        """Appends text to the log window."""
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.config(state="disabled")
        self.output_text.see(tk.END)

    def display_image(self, path):
        """Displays the final result image in the GUI."""
        img = Image.open(path)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.result_label.config(image=img_tk)
        self.result_label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = JigsawGUI(root)
    root.mainloop()