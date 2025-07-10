# app.py (Final Professional Version)

import customtkinter as ctk
from tkinter import filedialog, Canvas, ttk, messagebox
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageGrab
import threading
import os
import numpy as np
import time
import pyperclip
from fpdf import FPDF
from predictor import XRayPredictor
import sqlite3
from datetime import datetime
import json

# --- CONFIGURATION ---
MODEL_PATH = 'shoulder_xray_model.h5'
MODEL_VERSION = 'v2.1.3'
SAVE_DIR = "F:/The Lsst-projects/Rehabilitation/final-code/SaveResults"
DB_PATH = "scan_history.db"
MODEL_INFO = {
    "architecture": "EfficientNetB1",
    "training_date": "2024-03-15",
    "dataset_size": "1,240 X-ray images",
    "accuracy": "92.4% (validation)",
    "classes": ["A1", "C1", "D1", "Others"]
}

# --- UI COLORS & FONTS ---
COLOR_ACCENT_ORANGE = "#FFA726"
COLOR_ACCENT_BLUE = "#2979FF"
COLOR_ACCENT_GREEN = "#00C853"
COLOR_CRITICAL = "#e53935"
COLOR_BACKGROUND = "#1c1f26"
COLOR_CARD = "#2a2e38"
COLOR_TEXT = "#e0e0e0"

class ToolTip:
    """Creates a tooltip for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip = ctk.CTkToplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = ctk.CTkLabel(self.tooltip, text=self.text, corner_radius=6, 
                            fg_color="#333", text_color="white", wraplength=200)
        label.pack(ipadx=5, ipady=5)

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class HistoryDialog(ctk.CTkToplevel):
    """Dialog window to display and manage scan history from the database."""
    def __init__(self, parent, db_path):
        super().__init__(parent)
        self.db_path = db_path
        self.parent = parent
        self.title("Scan History")
        self.geometry("800x600")
        self.transient(parent)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Search bar
        search_frame = ctk.CTkFrame(self)
        search_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search by filename or type...")
        self.search_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.search_entry.bind("<Return>", lambda event: self.load_history())
        
        search_btn = ctk.CTkButton(search_frame, text="🔍", width=30, command=self.load_history)
        search_btn.pack(side="left", padx=5)
        
        # Treeview for history
        self.tree = ttk.Treeview(self, columns=("date", "type", "confidence", "file"), show="headings")
        self.tree.heading("date", text="Date")
        self.tree.heading("type", text="Glenoid Type")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("file", text="File Path")
        self.tree.column("date", width=150, anchor="w")
        self.tree.column("type", width=100, anchor="center")
        self.tree.column("confidence", width=100, anchor="center")
        self.tree.column("file", width=400, anchor="w")
        
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(0,10))
        
        self.tree.bind("<Double-1>", lambda event: self.open_selected())

        # Bottom buttons
        btn_frame = ctk.CTkFrame(self)
        btn_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        open_btn = ctk.CTkButton(btn_frame, text="Open Selected", command=self.open_selected)
        open_btn.pack(side="left", padx=5)
        
        delete_btn = ctk.CTkButton(btn_frame, text="Delete Selected", fg_color=COLOR_CRITICAL, 
                                  hover_color="#c62828", command=self.delete_selected)
        delete_btn.pack(side="left", padx=5)
        
        close_btn = ctk.CTkButton(btn_frame, text="Close", command=self.destroy)
        close_btn.pack(side="right", padx=5)
        
        self.load_history()
    
    def load_history(self):
        """Loads scan history from the database into the Treeview."""
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            search_term = self.search_entry.get()
            query = "SELECT timestamp, glenoid_type, confidence, filename FROM scans "
            params = []
            if search_term:
                query += "WHERE filename LIKE ? OR glenoid_type LIKE ? "
                params.extend([f"%{search_term}%", f"%{search_term}%"])
            query += "ORDER BY timestamp DESC"
            
            cursor.execute(query, tuple(params))
            for row in cursor.fetchall():
                self.tree.insert("", "end", values=(
                    datetime.fromtimestamp(row[0]).strftime("%Y-%m-%d %H:%M"),
                    row[1], f"{float(row[2]):.2f}%", row[3]
                ))

    def open_selected(self):
        """Opens the selected image from history in the main app."""
        selected_item = self.tree.selection()
        if not selected_item:
            return
        
        filepath = self.tree.item(selected_item[0], "values")[3]
        self.parent.open_history_item(filepath)
        self.destroy()
    
    def delete_selected(self):
        """Deletes the selected record from the database."""
        selected_item = self.tree.selection()
        if not selected_item:
            return

        filepath = self.tree.item(selected_item[0], "values")[3]
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the history record for:\n{os.path.basename(filepath)}?"):
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM scans WHERE filename=?", (filepath,))
            self.load_history()

class ModelInfoDialog(ctk.CTkToplevel):
    """Dialog window to display detailed AI model information."""
    def __init__(self, parent, model_info):
        super().__init__(parent)
        self.title("Model Information")
        self.geometry("500x400")
        self.transient(parent)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self, text="AI Model Specifications", font=ctk.CTkFont(size=16, weight="bold")).pack(
            padx=20, pady=10, anchor="w")
        
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(fill="x", expand=True, padx=20, pady=10)
        info_frame.grid_columnconfigure(1, weight=1)
        
        for i, (key, value) in enumerate(model_info.items()):
            ctk.CTkLabel(info_frame, text=f"{key.replace('_', ' ').title()}:", font=ctk.CTkFont(weight="bold")).grid(
                row=i, column=0, padx=10, pady=5, sticky="w")
            ctk.CTkLabel(info_frame, text=value, wraplength=300, justify="left").grid(
                row=i, column=1, padx=10, pady=5, sticky="w")
        
        ctk.CTkButton(self, text="Close", command=self.destroy).pack(pady=20)

class App(ctk.CTk):
    """Main application class for the AI Shoulder X-ray Analysis Dashboard."""
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.title("AI Shoulder X-ray Analysis Dashboard")
        self.geometry("1600x900")
        
        # --- ENHANCEMENT: Ensure save directory exists on startup ---
        if not os.path.exists(SAVE_DIR):
            try:
                os.makedirs(SAVE_DIR)
                print(f"Created save directory: {SAVE_DIR}")
            except OSError as e:
                print(f"Error creating directory {SAVE_DIR}: {e}")
                messagebox.showerror("Directory Error", f"Could not create save directory:\n{SAVE_DIR}\nPlease check permissions.")

        self.init_db()
        
        # State variables
        self.image_path, self.original_image, self.processed_image, self.last_results = None, None, None, {}
        self.ctk_image = None # Reference to the displayed CTkImage object
        self.zoom_level, self.brightness_level = 1.0, 1.0
        self.edge_var, self.invert_var, self.highlight_var = ctk.BooleanVar(), ctk.BooleanVar(), ctk.BooleanVar()
        self.mode_switch_var = ctk.StringVar(value="on")
        self.dominant_side_var = ctk.StringVar(value="Right")
        
        self.configure(fg_color=COLOR_BACKGROUND)
        
        # Layout configuration
        self.grid_columnconfigure(0, weight=2, minsize=320)
        self.grid_columnconfigure(1, weight=5)
        self.grid_columnconfigure(2, weight=3, minsize=400)
        self.grid_rowconfigure(0, weight=1)

        # Build UI components
        self.create_left_panel()
        self.create_middle_panel()
        self.create_right_panel()
        self.clear_all()
        ctk.set_appearance_mode("dark")
    
    def init_db(self):
        """Initializes the SQLite database and creates the 'scans' table if it doesn't exist."""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS scans
                             (id INTEGER PRIMARY KEY AUTOINCREMENT,
                              timestamp REAL,
                              glenoid_type TEXT,
                              confidence REAL,
                              filename TEXT UNIQUE,
                              metadata TEXT)''')
    
    def save_to_history(self):
        """Saves the current analysis result to the database."""
        if not self.image_path or "error" in self.last_results:
            return
            
        # ENHANCEMENT: Store absolute path for reliability
        abs_image_path = os.path.abspath(self.image_path)
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            metadata = {
                "patient_info": {
                    "age": self.age_entry.get(),
                    "gender": self.gender_entry.get(),
                    "dominant_side": self.dominant_side_var.get(),
                    "symptoms": self.symptoms_entry.get()
                },
                "results": self.last_results
            }
            # Use INSERT OR REPLACE to update the record if the file is re-analyzed
            cursor.execute('''INSERT OR REPLACE INTO scans 
                             (timestamp, glenoid_type, confidence, filename, metadata)
                             VALUES (?, ?, ?, ?, ?)''',
                          (time.time(),
                           self.last_results.get('glenoid_type', 'Unknown'),
                           float(self.last_results.get('confidence', '0').replace('%', '')),
                           abs_image_path,
                           json.dumps(metadata)))
    
    def open_history_item(self, filepath):
        """Loads an image and its associated data from a history record."""
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"The file could not be found at the recorded path:\n{filepath}")
            return
            
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metadata FROM scans WHERE filename=?", (filepath,))
            result = cursor.fetchone()
        
        if result:
            self.import_image(filepath) # Import the image first
            metadata = json.loads(result[0])
            
            p_info = metadata.get('patient_info', {})
            self.age_entry.delete(0, 'end'); self.age_entry.insert(0, p_info.get('age', ''))
            self.gender_entry.delete(0, 'end'); self.gender_entry.insert(0, p_info.get('gender', ''))
            self.dominant_side_var.set(p_info.get('dominant_side', 'Right'))
            self.symptoms_entry.delete(0, 'end'); self.symptoms_entry.insert(0, p_info.get('symptoms', ''))
            
            self.last_results = metadata.get('results', {})
            # Update UI with results (duration is not stored, so pass 0)
            self.update_ui_with_results(self.last_results, 0)
    
    def create_left_panel(self):
        """Creates the left panel containing upload controls and patient info."""
        self.left_panel = ctk.CTkFrame(self, corner_radius=10, fg_color=COLOR_CARD)
        self.left_panel.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        self.left_panel.grid_rowconfigure(6, weight=1)
        
        ctk.CTkLabel(self.left_panel, text="📁 Upload Shoulder X-ray", 
                    font=ctk.CTkFont(size=20, weight="bold")).grid(
                        row=0, column=0, columnspan=2, padx=20, pady=20, sticky="w")
        
        ctk.CTkButton(self.left_panel, text="Import Image", command=self.import_image, 
                      fg_color=COLOR_ACCENT_BLUE, height=40, hover_color="#1c5fd1").grid(
                          row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        
        ctk.CTkButton(self.left_panel, text="📋 Scan History", command=self.show_history,
                     fg_color="transparent", border_width=2).grid(
                         row=2, column=0, columnspan=2, padx=20, pady=(0,10), sticky="ew")
        
        meta_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        meta_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        meta_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(meta_frame, text="Filename:").grid(row=0, column=0, sticky="w", padx=(0,5))
        self.filename_label = ctk.CTkLabel(meta_frame, text="", wraplength=200, anchor="w")
        self.filename_label.grid(row=0, column=1, sticky="ew")
        ctk.CTkLabel(meta_frame, text="Resolution:").grid(row=1, column=0, sticky="w", padx=(0,5))
        self.resolution_label = ctk.CTkLabel(meta_frame, text="", anchor="w")
        self.resolution_label.grid(row=1, column=1, sticky="ew")
        
        self.analyze_button = ctk.CTkButton(
            self.left_panel, text="🔘 Analyze Image", command=self.analyze_image_thread,
            fg_color=COLOR_ACCENT_GREEN, height=40, hover_color="#009944")
        self.analyze_button.grid(row=4, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        
        self.clear_button = ctk.CTkButton(
            self.left_panel, text="🧹 Clear All", command=self.clear_all,
            fg_color="transparent", border_width=2)
        self.clear_button.grid(row=5, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        
        self.patient_info_container = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.patient_info_container.grid(row=6, column=0, columnspan=2, padx=20, pady=10, sticky="new")
        self.patient_info_container.grid_remove()
        
        ctk.CTkLabel(self.patient_info_container, text="🧍 Patient Info", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(0,5))
        info_grid = ctk.CTkFrame(self.patient_info_container, fg_color="transparent")
        info_grid.pack(fill="x"); info_grid.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(info_grid, text="Age:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.age_entry = ctk.CTkEntry(info_grid); self.age_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ctk.CTkLabel(info_grid, text="Gender:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.gender_entry = ctk.CTkEntry(info_grid, placeholder_text="M/F/Other"); self.gender_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ctk.CTkLabel(info_grid, text="Dominant Side:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        side_frame = ctk.CTkFrame(info_grid, fg_color="transparent"); side_frame.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        ctk.CTkRadioButton(side_frame, text="Left", variable=self.dominant_side_var, value="Left").pack(side="left")
        ctk.CTkRadioButton(side_frame, text="Right", variable=self.dominant_side_var, value="Right").pack(side="left", padx=10)
        ctk.CTkLabel(info_grid, text="Symptoms:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.symptoms_entry = ctk.CTkEntry(info_grid, placeholder_text="Pain, limited mobility, etc."); self.symptoms_entry.grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        
        bottom_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        bottom_frame.grid(row=7, column=0, columnspan=2, padx=20, pady=10, sticky="sew")
        bottom_frame.grid_columnconfigure((0,1), weight=1)
        self.toggle_patient_btn = ctk.CTkButton(bottom_frame, text="Patient Info ▼", command=self.toggle_patient_info)
        self.toggle_patient_btn.grid(row=0, column=0, sticky="sw")
        ctk.CTkSwitch(bottom_frame, text="Dark Mode", command=self.toggle_dark_mode,
                     variable=self.mode_switch_var, onvalue="on", offvalue="off").grid(row=0, column=1, sticky="se")

        ctk.CTkButton(self.left_panel, text="ℹ️ Model Info", command=self.show_model_info).grid(
            row=8, column=0, columnspan=2, padx=20, pady=(0,10), sticky="ew")

    def create_middle_panel(self):
        """Creates the middle panel containing the image viewer and tools."""
        self.middle_panel = ctk.CTkFrame(self, corner_radius=10, fg_color=COLOR_CARD)
        self.middle_panel.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")
        self.middle_panel.grid_rowconfigure(1, weight=1)
        self.middle_panel.grid_columnconfigure(0, weight=1)
        
        tools_frame = ctk.CTkFrame(self.middle_panel, fg_color="transparent")
        tools_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(tools_frame, text="Zoom:").pack(side="left", padx=5)
        ctk.CTkButton(tools_frame, text="-", width=30, command=lambda: self.update_zoom(0.9)).pack(side="left")
        ctk.CTkButton(tools_frame, text="+", width=30, command=lambda: self.update_zoom(1.1)).pack(side="left", padx=5)
        
        ctk.CTkLabel(tools_frame, text="Brightness:").pack(side="left", padx=10)
        self.brightness_slider = ctk.CTkSlider(tools_frame, from_=0.5, to=1.5, command=self.update_brightness)
        self.brightness_slider.pack(side="left", fill="x", expand=True, padx=5)
        
        self.viewer_frame = ctk.CTkFrame(self.middle_panel, border_width=1, fg_color="#1a1a1a")
        self.viewer_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.viewer_frame.grid_propagate(False)
        self.viewer_frame.grid_rowconfigure(0, weight=1)
        self.viewer_frame.grid_columnconfigure(0, weight=1)
        self.image_label = ctk.CTkLabel(self.viewer_frame, text=""); self.image_label.grid(row=0, column=0)
        
        adv_tools_frame = ctk.CTkFrame(self.middle_panel, fg_color="transparent")
        adv_tools_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        sw1 = ctk.CTkSwitch(adv_tools_frame, text="Edge Detection", variable=self.edge_var, command=self.apply_filters_and_display)
        sw1.pack(side="left", padx=10); ToolTip(sw1, "Enhances edges in the X-ray for better visualization")
        sw2 = ctk.CTkSwitch(adv_tools_frame, text="Invert Contrast", variable=self.invert_var, command=self.apply_filters_and_display)
        sw2.pack(side="left", padx=10); ToolTip(sw2, "Inverts the colors of the X-ray (negative view)")
        sw3 = ctk.CTkSwitch(adv_tools_frame, text="Highlight Bones", variable=self.highlight_var, command=self.apply_filters_and_display)
        sw3.pack(side="left", padx=10); ToolTip(sw3, "Enhances contrast to better visualize bone structures")

    def create_right_panel(self):
        """Creates the right panel containing results and export options."""
        self.right_panel = ctk.CTkScrollableFrame(self, fg_color=COLOR_CARD)
        self.right_panel.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="nsew")
        
        result_card = ctk.CTkFrame(self.right_panel, fg_color=COLOR_CARD); result_card.pack(fill="x", expand=True, pady=(0, 10))
        ctk.CTkLabel(result_card, text="🧠 Classification Result", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, columnspan=2, padx=15, pady=15, sticky="w")
        self.confidence_canvas = Canvas(result_card, width=100, height=100, bg=COLOR_CARD, highlightthickness=0); self.confidence_canvas.grid(row=3, column=0, padx=15, pady=5, sticky="w")
        self.draw_confidence_meter(0)
        ctk.CTkLabel(result_card, text="Glenoid Type:").grid(row=1, column=1, sticky="sw", padx=10)
        self.glenoid_label = ctk.CTkLabel(result_card, text="", font=ctk.CTkFont(size=16, weight="bold")); self.glenoid_label.grid(row=2, column=1, sticky="nw", padx=10)
        ToolTip(self.glenoid_label, "Classification of glenoid morphology according to the Walch classification system")
        ctk.CTkLabel(result_card, text="Confidence:").grid(row=3, column=1, sticky="sw", padx=10)
        self.confidence_interp_label = ctk.CTkLabel(result_card, text="", font=ctk.CTkFont(weight="bold")); self.confidence_interp_label.grid(row=4, column=1, sticky="nw", padx=10)
        ctk.CTkLabel(result_card, text="Condition:").grid(row=5, column=0, padx=15, pady=(10,0), sticky="w")
        self.condition_label = ctk.CTkLabel(result_card, text="", wraplength=350, justify="left"); self.condition_label.grid(row=6, column=0, columnspan=2, padx=15, pady=(0,10), sticky="w")
        ctk.CTkLabel(result_card, text=f"Model: {MODEL_VERSION}").grid(row=7, column=0, padx=15, pady=(0,10), sticky="w")
        self.time_label = ctk.CTkLabel(result_card, text=""); self.time_label.grid(row=7, column=1, padx=15, pady=(0,10), sticky="e")
        model_label = ctk.CTkLabel(result_card, text="ℹ️", cursor="hand2"); model_label.grid(row=0, column=1, sticky="e", padx=15); model_label.bind("<Button-1>", lambda e: self.show_model_info()); ToolTip(model_label, "Click for detailed model information")
        
        reco_card = ctk.CTkFrame(self.right_panel, fg_color=COLOR_CARD); reco_card.pack(fill="x", expand=True, pady=10)
        ctk.CTkLabel(reco_card, text="🔩 Recommended Prosthetic", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=15, pady=15)
        self.prosthetic_label = ctk.CTkLabel(reco_card, text="", wraplength=350, justify="left"); self.prosthetic_label.pack(anchor="w", padx=15, pady=(0,15)); ToolTip(self.prosthetic_label, "Prosthetic recommendation based on glenoid classification")
        
        export_card = ctk.CTkFrame(self.right_panel, fg_color=COLOR_CARD); export_card.pack(fill="x", expand=True, pady=10)
        ctk.CTkLabel(export_card, text="📤 Export", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=15, pady=15)
        self.pdf_button = ctk.CTkButton(export_card, text="📄 Download PDF Report", command=self.download_pdf_report, fg_color=COLOR_ACCENT_BLUE, hover_color="#1c5fd1"); self.pdf_button.pack(fill="x", padx=15, pady=5); ToolTip(self.pdf_button, "Generate a comprehensive PDF report with all findings")
        self.ss_button = ctk.CTkButton(export_card, text="📸 Export Screenshot", command=self.export_screenshot, fg_color=COLOR_ACCENT_BLUE, hover_color="#1c5fd1"); self.ss_button.pack(fill="x", padx=15, pady=5); ToolTip(self.ss_button, "Save a screenshot of the current analysis")
        self.copy_button = ctk.CTkButton(export_card, text="🔗 Copy to Clipboard", command=self.copy_to_clipboard, fg_color=COLOR_ACCENT_BLUE, hover_color="#1c5fd1"); self.copy_button.pack(fill="x", padx=15, pady=5); ToolTip(self.copy_button, "Copy analysis results to clipboard for pasting into other applications")
        self.share_button = ctk.CTkButton(export_card, text="🌐 Share to Doctor Portal", command=self.share_to_portal, fg_color=COLOR_ACCENT_GREEN, hover_color="#009944"); self.share_button.pack(fill="x", padx=15, pady=(5,15)); ToolTip(self.share_button, "Upload results to the hospital's doctor portal (placeholder)")

    def draw_confidence_meter(self, percentage):
        """Draws the circular confidence meter on the canvas."""
        self.confidence_canvas.delete("all")
        w, h = 100, 100
        radius = w // 2 - 10
        center_x, center_y = w // 2, h // 2
        self.confidence_canvas.create_oval(center_x-radius, center_y-radius, center_x+radius, center_y+radius, outline="#444", width=4)
        if percentage > 0:
            color = COLOR_ACCENT_GREEN if percentage >= 90 else COLOR_ACCENT_ORANGE if percentage >= 70 else COLOR_CRITICAL
            extent = 360 * (percentage / 100)
            self.confidence_canvas.create_arc(center_x-radius, center_y-radius, center_x+radius, center_y+radius, start=90, extent=-extent, outline=color, width=8, style="arc")
        self.confidence_canvas.create_text(center_x, center_y, text=f"{percentage:.0f}%", font=("Arial", 16, "bold"), fill=COLOR_TEXT)
    
    def show_history(self):
        HistoryDialog(self, DB_PATH)
    
    def show_model_info(self):
        ModelInfoDialog(self, MODEL_INFO)
    
    def share_to_portal(self):
        # ENHANCEMENT: Use a professional-looking messagebox for placeholder function
        messagebox.showinfo(
            "Portal Share (Placeholder)",
            "This feature is a placeholder for future integration.\n\n"
            "In a real-world scenario, this would trigger an API call to securely "
            "upload the analysis report to a hospital's doctor portal or EMR system."
        )

    def import_image(self, filepath=None):
        """Imports, processes, and displays a new image dynamically."""
        if not filepath:
            filepath = filedialog.askopenfilename(title="Select an X-ray Image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.dcm")])
        if not filepath: return

        try:
            # --- FIX: Don't call clear_all(). We are replacing the image, not clearing the entire UI. ---
            # 1. Reset only the previous results, not the entire UI state
            self.last_results = {}
            self.glenoid_label.configure(text=""); self.condition_label.configure(text=""); self.prosthetic_label.configure(text="")
            self.time_label.configure(text=""); self.confidence_interp_label.configure(text=""); self.draw_confidence_meter(0)
            for btn in [self.pdf_button, self.ss_button, self.copy_button, self.share_button]: btn.configure(state="disabled")

            # 2. Load the new image
            self.original_image = Image.open(filepath).convert("RGB")
            self.image_path = filepath
            self.zoom_level = 1.0 # Reset zoom for new image
            
            # 3. Update metadata and button states
            self.filename_label.configure(text=os.path.basename(filepath))
            self.resolution_label.configure(text=f"{self.original_image.width}x{self.original_image.height}")
            
            self.analyze_button.configure(state="normal")
            self.clear_button.configure(state="normal")
            
            # 4. Process and display the new image
            self.apply_filters_and_display()

        except Exception as e:
            # If loading fails, then clear everything to a clean state
            self.clear_all() 
            self.image_label.configure(text=f"Error: Cannot open image file.\n{e}")
            messagebox.showerror("Image Error", f"Could not load the selected image.\n\nError: {e}")

    # PERFORMANCE ENHANCEMENT: Separated filter application from display resizing
    def apply_filters_and_display(self):
        """Applies all selected filters to the original image and then calls display."""
        if not self.original_image: return

        self.processed_image = self.original_image.copy()
        if self.invert_var.get(): self.processed_image = ImageOps.invert(self.processed_image)
        if self.edge_var.get(): self.processed_image = self.processed_image.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
        if self.highlight_var.get():
            enhancer = ImageEnhance.Contrast(self.processed_image)
            self.processed_image = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Brightness(self.processed_image)
        self.processed_image = enhancer.enhance(self.brightness_level)
        
        self.display_image() # Now call display which only handles zoom/fit

    def display_image(self):
        """Handles resizing and displaying the processed image in the viewer."""
        if not self.processed_image:
            self.image_label.configure(image=None, text="Import an X-ray image to begin")
            return

        self.viewer_frame.update_idletasks()
        max_w = self.viewer_frame.winfo_width()
        max_h = self.viewer_frame.winfo_height()
        if max_w < 20 or max_h < 20: return

        img_to_show = self.processed_image.copy()
        
        # Apply zoom
        zoomed_size = (int(img_to_show.width * self.zoom_level), int(img_to_show.height * self.zoom_level))
        img_to_show = img_to_show.resize(zoomed_size, Image.LANCZOS)

        # Fit to viewer frame without distorting aspect ratio
        img_to_show.thumbnail((max_w - 10, max_h - 10), Image.LANCZOS)
        
        self.ctk_image = ctk.CTkImage(light_image=img_to_show, dark_image=img_to_show, size=img_to_show.size)
        self.image_label.configure(image=self.ctk_image, text="")
        
    def analyze_image_thread(self):
        """Starts the image analysis in a separate thread to avoid freezing the UI."""
        if self.image_path:
            threading.Thread(target=self.run_prediction, daemon=True).start()

    def run_prediction(self):
        """Executes the prediction and updates the UI via the main thread."""
        self.after(0, lambda: self.analyze_button.configure(state="disabled", text="Analyzing..."))
        start_time = time.time()
        self.last_results = self.predictor.predict(self.image_path)
        duration = time.time() - start_time
        self.save_to_history()
        self.after(0, self.update_ui_with_results, self.last_results, duration)

    def update_ui_with_results(self, results, duration):
        """Updates the right panel with the analysis results."""
        self.analyze_button.configure(state="normal", text="🔘 Analyze Image")
        
        if "error" in results:
            self.glenoid_label.configure(text="Error", text_color=COLOR_CRITICAL)
            self.condition_label.configure(text=results['error'])
            return

        glenoid_type = results.get('glenoid_type', 'N/A')
        confidence = float(results.get('confidence', '0').replace('%', ''))
        
        color_map = {"A1": COLOR_ACCENT_GREEN, "C1": COLOR_ACCENT_ORANGE, "D1": COLOR_CRITICAL, "Others": "#a2a2a2"}
        self.glenoid_label.configure(text=glenoid_type, text_color=color_map.get(glenoid_type, COLOR_TEXT))
        self.condition_label.configure(text=results.get('condition', 'N/A'))
        self.prosthetic_label.configure(text=results.get('prosthetic', 'N/A'))
        if duration > 0: self.time_label.configure(text=f"Time: {duration:.2f} sec")
        self.draw_confidence_meter(confidence)
        
        if confidence >= 90: self.confidence_interp_label.configure(text="✅ High Confidence", text_color=COLOR_ACCENT_GREEN)
        elif confidence < 70: self.confidence_interp_label.configure(text="⚠️ Review Recommended", text_color=COLOR_ACCENT_ORANGE)
        else: self.confidence_interp_label.configure(text="Moderate Confidence", text_color=COLOR_TEXT)
        
        for btn in [self.pdf_button, self.ss_button, self.copy_button, self.share_button]: btn.configure(state="normal")
            
    def clear_all(self):
        """Resets the entire application to its initial state."""
        self.image_path, self.original_image, self.processed_image, self.last_results = None, None, None, {}
        # --- FIX: Explicitly clear the CTkImage object reference to prevent display bugs ---
        self.ctk_image = None
        
        self.zoom_level, self.brightness_level = 1.0, 1.0
        self.edge_var.set(False); self.invert_var.set(False); self.highlight_var.set(False)
        self.brightness_slider.set(1.0)
        
        self.filename_label.configure(text="N/A")
        self.resolution_label.configure(text="N/A")
        
        self.age_entry.delete(0, 'end'); self.gender_entry.delete(0, 'end'); self.symptoms_entry.delete(0, 'end')
        self.dominant_side_var.set("Right")
        
        self.analyze_button.configure(state="disabled")
        self.clear_button.configure(state="disabled")
        
        self.glenoid_label.configure(text=""); self.condition_label.configure(text=""); self.prosthetic_label.configure(text="")
        self.time_label.configure(text=""); self.confidence_interp_label.configure(text=""); self.draw_confidence_meter(0)
        
        for btn in [self.pdf_button, self.ss_button, self.copy_button, self.share_button]: btn.configure(state="disabled")
        
        # This will now correctly show the placeholder text because self.processed_image is None
        self.display_image()

    def toggle_dark_mode(self):
        ctk.set_appearance_mode("Dark" if self.mode_switch_var.get() == "on" else "Light")
    
    def update_zoom(self, factor):
        if self.processed_image: 
            self.zoom_level = max(0.1, self.zoom_level * factor) # Prevent zoom from becoming zero or negative
            self.display_image() # Only call display, no need to re-apply filters
    
    def update_brightness(self, value):
        if self.original_image: 
            self.brightness_level = float(value)
            self.apply_filters_and_display() # Re-apply all filters with new brightness
    
    def toggle_patient_info(self):
        if self.patient_info_container.winfo_viewable(): 
            self.patient_info_container.grid_remove()
            self.toggle_patient_btn.configure(text="Patient Info ▼")
        else: 
            self.patient_info_container.grid()
            self.toggle_patient_btn.configure(text="Patient Info ▲")

    def save_file_dialog(self, content, extension, mode='w'):
        """Opens a 'save as' dialog and saves content to the selected file."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        initial_filename = f"analysis_{os.path.basename(self.image_path or 'image')}_{timestamp}.{extension}"
        
        save_path = filedialog.asksaveasfilename(
            initialdir=SAVE_DIR, initialfile=initial_filename, defaultextension=f".{extension}",
            filetypes=[(f"{extension.upper()} files", f"*.{extension}"), ("All files", "*.*")])
        if not save_path: return

        try:
            if isinstance(content, Image.Image):
                content.save(save_path)
            elif mode == 'wb':
                with open(save_path, 'wb') as f: f.write(content)
            else:
                with open(save_path, 'w', encoding='utf-8') as f: f.write(content)
            messagebox.showinfo("Success", f"File saved successfully to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save the file.\n\nError: {e}")

    def download_pdf_report(self):
        """Generates and saves a PDF report of the analysis."""
        if not self.last_results or 'error' in self.last_results: return
            
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, 'AI Shoulder X-ray Analysis Report', 0, 1, 'C'); pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Patient Information", "B", 1, 'L'); pdf.ln(4)
        patient_info = {"Age": self.age_entry.get(), "Gender": self.gender_entry.get(), "Dominant Side": self.dominant_side_var.get(), "Symptoms": self.symptoms_entry.get()}
        for key, value in patient_info.items():
            if value:
                pdf.set_font("Arial", 'B', 11); pdf.cell(40, 8, f"{key}:", 0, 0)
                pdf.set_font("Arial", '', 11); pdf.multi_cell(0, 8, value, 0, 1)
        
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Analysis Results", "B", 1, 'L'); pdf.ln(4)
        for key, value in self.last_results.items():
            pdf.set_font("Arial", 'B', 11); pdf.cell(50, 8, f"{key.replace('_', ' ').title()}:", 0, 0)
            pdf.set_font("Arial", '', 11); pdf.multi_cell(0, 8, str(value), 0, 1)
        
        if self.original_image:
            pdf.ln(10)
            temp_img_path = os.path.join(SAVE_DIR, "temp_report_img.jpg")
            try:
                self.original_image.save(temp_img_path)
                pdf.image(temp_img_path, x=10, w=pdf.w - 20)
                os.remove(temp_img_path)
            except Exception as e:
                print(f"Could not add image to PDF: {e}")
        
        pdf.set_y(-25)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, f"Generated by AI Analysis {MODEL_VERSION} on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0, 'C')
        
        self.save_file_dialog(pdf.output(dest='S').encode('latin-1'), 'pdf', 'wb')

    def export_screenshot(self):
        """Captures and saves a screenshot of the entire application window."""
        try:
            x = self.winfo_rootx(); y = self.winfo_rooty()
            w = self.winfo_width(); h = self.winfo_height()
            screenshot = ImageGrab.grab(bbox=(x, y, x+w, y+h), all_screens=True)
            self.save_file_dialog(screenshot, 'png')
        except Exception as e: 
            messagebox.showerror("Screenshot Error", f"Could not capture the screen.\n\nError: {e}")

    def copy_to_clipboard(self):
        """Copies a text summary of the analysis to the clipboard."""
        if not self.last_results or 'error' in self.last_results: return
            
        report = "--- AI Analysis Summary ---\n"
        patient_info = {"Age": self.age_entry.get(), "Gender": self.gender_entry.get(), "Dominant Side": self.dominant_side_var.get(), "Symptoms": self.symptoms_entry.get()}
        if any(patient_info.values()):
            report += "\nPatient Information:\n"
            for key, value in patient_info.items():
                if value: report += f"- {key}: {value}\n"
        
        report += "\nAnalysis Results:\n"
        for key, value in self.last_results.items(): report += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        report += f"\nModel Version: {MODEL_VERSION}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        pyperclip.copy(report)
        messagebox.showinfo("Success", "Report summary copied to clipboard.")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH): 
        messagebox.showerror("Fatal Error", f"Model file not found: {MODEL_PATH}\nThe application will now exit.")
        exit()
    
    predictor = XRayPredictor(model_path=MODEL_PATH)
    if not predictor.model: 
        messagebox.showerror("Fatal Error", "The AI model failed to load.\nPlease check the model file and dependencies.\nThe application will now exit.")
        exit()
    
    app = App(predictor)
    
    # Model Warm-up (optional but improves first prediction speed)
    print("Warming up model...")
    try:
        predictor.model.predict(np.zeros((1, 256, 256, 3), dtype=np.float32), verbose=0)
        print("Model is ready.")
    except Exception as e:
        print(f"Model warm-up failed: {e}")

    app.mainloop()