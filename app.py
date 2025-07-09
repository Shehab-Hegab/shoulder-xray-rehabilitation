# app.py (Version 6.0 - Final Professional Edition with Model Warm-up)

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import threading
import os
import numpy as np # <-- IMPORT NUMPY FOR THE DUMMY IMAGE

# Import the custom predictor class
from predictor import XRayPredictor

# --- CONFIGURATION ---
MODEL_PATH = 'shoulder_xray_model.h5'
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

class App(ctk.CTk):
    def __init__(self, predictor):
        super().__init__()
        
        self.predictor = predictor
        
        self.title("AI Shoulder X-ray Classification")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create all the user interface widgets
        self.create_widgets()

    def create_widgets(self):
        """Creates and places all the widgets in the window."""
        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Prosthetic Selection", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.import_button = ctk.CTkButton(self.sidebar_frame, text="Import Image", command=self.select_image_thread)
        self.import_button.grid(row=1, column=0, padx=20, pady=10)
        
        # --- Main Content Frame ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1, minsize=350)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # --- Image Display Frame ---
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="Import an X-ray image to begin analysis", font=ctk.CTkFont(size=16))
        self.image_label.grid(row=0, column=0, padx=10, pady=10)

        # --- Results Frame ---
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Classified Condition Section
        self.condition_header = ctk.CTkLabel(self.results_frame, text="Classified Condition", font=ctk.CTkFont(size=18, weight="bold"))
        self.condition_header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        self.glenoid_type_label = ctk.CTkLabel(self.results_frame, text="Glenoid Type: N/A", font=ctk.CTkFont(size=22, weight="bold"), text_color="#5DADE2")
        self.glenoid_type_label.grid(row=1, column=0, padx=20, pady=(10,5), sticky="w")
        
        self.condition_label = ctk.CTkLabel(self.results_frame, text="Condition: N/A", font=ctk.CTkFont(size=14))
        self.condition_label.grid(row=2, column=0, padx=20, pady=5, sticky="w")

        # Confidence Score Section
        self.confidence_header = ctk.CTkLabel(self.results_frame, text="Confidence Score", font=ctk.CTkFont(size=14))
        self.confidence_header.grid(row=3, column=0, padx=20, pady=(20, 5), sticky="w")
        
        self.confidence_bar = ctk.CTkProgressBar(self.results_frame, orientation="horizontal")
        self.confidence_bar.set(0)
        self.confidence_bar.grid(row=4, column=0, padx=20, pady=0, sticky="ew")

        self.confidence_label = ctk.CTkLabel(self.results_frame, text="0.00%", font=ctk.CTkFont(size=12))
        self.confidence_label.grid(row=5, column=0, padx=20, pady=(0, 20), sticky="e")

        # Divider
        self.divider = ctk.CTkFrame(self.results_frame, height=2)
        self.divider.grid(row=6, column=0, padx=20, pady=10, sticky="ew")

        # Recommended Prosthetic Section
        self.prosthetic_header = ctk.CTkLabel(self.results_frame, text="Recommended Prosthetic", font=ctk.CTkFont(size=18, weight="bold"))
        self.prosthetic_header.grid(row=7, column=0, padx=20, pady=10, sticky="w")

        self.prosthetic_label = ctk.CTkLabel(self.results_frame, text="N/A", font=ctk.CTkFont(size=14), wraplength=300, justify="left")
        self.prosthetic_label.grid(row=8, column=0, padx=20, pady=5, sticky="w")
    
    def select_image_thread(self):
        """Handles the 'Import Image' button click and starts the prediction in a new thread."""
        filepath = filedialog.askopenfilename(filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")))
        if filepath:
            # Run the heavy prediction task in a separate thread to keep the UI responsive
            thread = threading.Thread(target=self.run_prediction, args=(filepath,), daemon=True)
            thread.start()
            
    def run_prediction(self, filepath):
        """
        This function runs in the background thread.
        It updates the UI to a 'processing' state, gets the prediction, and updates again with the results.
        """
        # Step 1: Update UI to show 'processing' state (using the safe 'after' method)
        self.after(0, self.set_ui_to_processing, filepath)
        
        # Step 2: Get prediction from the predictor class (this is the potentially slow part on first run)
        results = self.predictor.predict(filepath)
        
        # Step 3: Schedule the final UI update on the main thread
        self.after(0, self.update_ui_with_results, results)

    def set_ui_to_processing(self, filepath):
        """This function runs on the main GUI thread and safely updates the UI."""
        self.import_button.configure(state="disabled", text="Processing...")
        self.display_image(filepath)
        self.glenoid_type_label.configure(text="Processing...", text_color="#5DADE2")
        self.condition_label.configure(text="")
        self.prosthetic_label.configure(text="")
        self.confidence_label.configure(text="...")
        self.confidence_bar.set(0)

    def update_ui_with_results(self, results):
        """This function also runs on the main GUI thread to display the final results."""
        if "error" in results:
            self.glenoid_type_label.configure(text=f"Error: {results['error']}", font=ctk.CTkFont(size=16, weight="bold"), text_color="#E74C3C")
        else:
            glenoid_type = results.get('glenoid_type', 'N/A')
            confidence_val = float(results.get('confidence', '0').replace('%', ''))
            
            # Dynamic Color Logic
            color_map = {"A1": "#2ECC71", "C1": "#F39C12", "D1": "#F39C12", "Others": "#E74C3C", "N/A": "#5DADE2"}
            result_color = color_map.get(glenoid_type, "#5DADE2")
            
            # Update UI elements with new data and colors
            self.glenoid_type_label.configure(text=f"Glenoid Type: {glenoid_type}", text_color=result_color)
            self.condition_label.configure(text=f"Condition: {results.get('condition', 'N/A')}")
            self.prosthetic_label.configure(text=results.get('prosthetic', 'N/A'))
            
            self.confidence_bar.configure(progress_color=result_color)
            self.confidence_bar.set(confidence_val / 100)
            self.confidence_label.configure(text=f"{confidence_val:.2f}%")
        
        self.import_button.configure(state="normal", text="Import Image")

    def display_image(self, filepath):
        """Loads and displays the selected image, fitting it to the frame."""
        try:
            frame_w, frame_h = 800, 600
            img = Image.open(filepath)
            img.thumbnail((frame_w, frame_h))
            ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.image_label.configure(image=ctk_image, text="")
        except Exception as e:
            self.image_label.configure(text=f"Error displaying image:\n{e}")

# =================================================================================================
# ## THE MOST IMPORTANT CHANGE FOR SPEED ##
# =================================================================================================
if __name__ == "__main__":
    # Check if the model file exists before launching the app
    if os.path.exists(MODEL_PATH):
        predictor = XRayPredictor(model_path=MODEL_PATH)
        
        if predictor.model:
            # --- MODEL WARM-UP ---
            # This is the trick to make the first user click fast.
            # We perform a "dummy" prediction right at the start.
            # This forces TensorFlow to load everything into memory and build its computation graph.
            # The user experiences this as a slightly longer app startup time,
            # but every subsequent prediction will be much faster.
            
            print("Warming up the model... (this may take a moment on first run)")
            
            # Create a dummy blank image with the correct shape (1, 256, 256, 3)
            # The model expects a batch of images, so the first dimension is 1.
            dummy_image_array = np.zeros((1, 256, 256, 3), dtype=np.float32)
            
            # Directly use the loaded model to predict on the dummy image.
            # We do this once to "wake up" the TensorFlow engine.
            predictor.model.predict(dummy_image_array, verbose=0) # verbose=0 hides the progress bar
            
            print("Model is ready and warmed up!")
            
            # Now that the model is warmed up, create and run the app
            app = App(predictor)
            app.mainloop()
        else:
            print("Model file found, but failed to load. It might be corrupted.")
    else:
        # If the model file is not found, print a helpful error message
        print(f"Error: Model file not found at path: {MODEL_PATH}")
        print("Please run 'python train_model.py' first to generate the model file.")

