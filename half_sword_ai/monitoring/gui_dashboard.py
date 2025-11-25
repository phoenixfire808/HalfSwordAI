"""
Python GUI Dashboard for Half Sword AI Agent
Replaces HTML dashboard with native Python window
Includes embedded YOLO overlay display
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import logging
import numpy as np
import cv2
from PIL import Image, ImageTk
from typing import Optional, Dict, Any
from half_sword_ai.config import config

logger = logging.getLogger(__name__)


class GUIDashboard:
    """Python GUI Dashboard with sliders and controls + embedded YOLO overlay"""
    
    def __init__(self, agent=None, performance_monitor=None, actor=None, learner=None):
        self.agent = agent
        self.performance_monitor = performance_monitor
        self.actor = actor
        self.learner = learner
        self.root = None
        self.running = False
        self.update_thread = None
        
        # YOLO overlay integration
        self.yolo_frame = None
        self.yolo_detections = {}
        self.yolo_lock = threading.Lock()
        self.yolo_label = None  # tkinter label for displaying YOLO feed
        
        # Store original config values for reset
        self.original_config = {}
        self._save_original_config()
        
        # Widget references
        self.widgets = {}
        self.stats_labels = {}
        
    def _save_original_config(self):
        """Save original config values"""
        attrs = [attr for attr in dir(config) if attr.isupper() and not attr.startswith('_')]
        for attr in attrs:
            try:
                self.original_config[attr] = getattr(config, attr)
            except:
                pass
    
    def start(self):
        """Start the GUI dashboard"""
        self.root = tk.Tk()
        self.root.title("Half Sword AI Agent - Control Panel")
        
        # Position window on right side of screen - single unified window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 1200
        window_height = screen_height - 50
        x_position = screen_width - window_width - 10  # 10px margin from right edge
        y_position = 25  # 25px from top
        
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        self.root.resizable(True, True)  # Allow resizing
        self.root.configure(bg='#1e1e1e')
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#1e1e1e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#2d2d2d', foreground='white', padding=[20, 10])
        style.map('TNotebook.Tab', background=[('selected', '#0078d4')])
        
        # Create unified single-window interface
        # Top section: YOLO overlay display (larger, prominent)
        top_frame = tk.Frame(self.root, bg='#1e1e1e')
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # YOLO display at top
        yolo_title = tk.Label(top_frame, text="YOLO Detection Overlay - Live Feed", 
                             font=('Arial', 16, 'bold'), bg='#1e1e1e', fg='#00ff00')
        yolo_title.pack(pady=(5, 10))
        
        yolo_display_frame = tk.Frame(top_frame, bg='#000000', relief=tk.SUNKEN, borderwidth=2)
        yolo_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.yolo_label = tk.Label(yolo_display_frame, bg='#000000', text="Waiting for frames...", 
                                  fg='white', font=('Arial', 10))
        self.yolo_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom section: Control tabs (compact)
        bottom_frame = tk.Frame(self.root, bg='#1e1e1e')
        bottom_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # Create notebook (tabs) for controls
        notebook = ttk.Notebook(bottom_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self._create_learning_tab(notebook)
        self._create_rewards_tab(notebook)
        self._create_input_tab(notebook)
        self._create_yolo_tab(notebook)
        self._create_performance_tab(notebook)
        self._create_control_tab(notebook)
        
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start GUI mainloop (runs in thread, non-blocking for main program)
        self.root.mainloop()
    
    def _create_learning_tab(self, notebook):
        """Create learning parameters tab"""
        frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(frame, text="Learning")
        
        canvas = tk.Canvas(frame, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1e1e1e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Learning Rate
        self._create_slider(scrollable_frame, "Learning Rate", "LEARNING_RATE", 
                          0.00001, 0.01, config.LEARNING_RATE, log_scale=True)
        
        # Gamma (Discount Factor)
        self._create_slider(scrollable_frame, "Gamma (Discount)", "GAMMA", 
                          0.5, 0.999, config.GAMMA)
        
        # Batch Size
        self._create_slider(scrollable_frame, "Batch Size", "BATCH_SIZE", 
                          1, 128, config.BATCH_SIZE, int_val=True)
        
        # Replay Buffer Size
        self._create_slider(scrollable_frame, "Replay Buffer Size", "REPLAY_BUFFER_SIZE", 
                          1000, 100000, config.REPLAY_BUFFER_SIZE, int_val=True)
        
        # Training Frequency
        self._create_slider(scrollable_frame, "Training Frequency (Hz)", "TRAINING_FREQUENCY", 
                          0.01, 1.0, config.TRAINING_FREQUENCY)
        
        # Update Frequency
        self._create_slider(scrollable_frame, "Update Frequency (batches)", "UPDATE_FREQUENCY", 
                          1, 20, config.UPDATE_FREQUENCY, int_val=True)
        
        # Priority Alpha
        self._create_slider(scrollable_frame, "Priority Alpha", "PRIORITY_ALPHA", 
                          0.0, 1.0, config.PRIORITY_ALPHA)
        
        # DQN Epsilon Start
        self._create_slider(scrollable_frame, "DQN Epsilon Start", "DQN_EPSILON_START", 
                          0.0, 1.0, config.DQN_EPSILON_START)
        
        # DQN Epsilon End
        self._create_slider(scrollable_frame, "DQN Epsilon End", "DQN_EPSILON_END", 
                          0.0, 0.1, config.DQN_EPSILON_END)
        
        # DQN Epsilon Decay
        self._create_slider(scrollable_frame, "DQN Epsilon Decay (frames)", "DQN_EPSILON_DECAY", 
                          10000, 5000000, config.DQN_EPSILON_DECAY, int_val=True)
        
        # DQN Target Update Frequency
        self._create_slider(scrollable_frame, "DQN Target Update Freq", "DQN_TARGET_UPDATE_FREQ", 
                          100, 10000, config.DQN_TARGET_UPDATE_FREQ, int_val=True)
        
        # Beta BC (Behavioral Cloning)
        self._create_slider(scrollable_frame, "Beta BC (Behavioral Cloning)", "BETA_BC", 
                          0.0, 10.0, config.BETA_BC)
        
        # Human Action Priority Multiplier
        self._create_slider(scrollable_frame, "Human Action Priority", "HUMAN_ACTION_PRIORITY_MULTIPLIER", 
                          1.0, 10.0, config.HUMAN_ACTION_PRIORITY_MULTIPLIER)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_rewards_tab(self, notebook):
        """Create rewards configuration tab"""
        frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(frame, text="Rewards")
        
        canvas = tk.Canvas(frame, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1e1e1e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable Comprehensive Rewards
        self._create_checkbox(scrollable_frame, "Enable Comprehensive Rewards", "ENABLE_COMPREHENSIVE_REWARDS")
        
        # Use Enhanced Rewards
        self._create_checkbox(scrollable_frame, "Use Enhanced Rewards", "USE_ENHANCED_REWARDS")
        
        # Enable PBRS
        self._create_checkbox(scrollable_frame, "Enable PBRS", "ENABLE_PBRS")
        
        # Reward Normalization
        self._create_checkbox(scrollable_frame, "Reward Normalization", "REWARD_NORMALIZATION")
        
        # Curriculum Phase
        self._create_dropdown(scrollable_frame, "Curriculum Phase", "CURRICULUM_PHASE",
                            ["toddler", "swordsman", "duelist", "master"])
        
        # Reward weights
        self._create_slider(scrollable_frame, "Reward: Survival", "REWARD_SURVIVAL", 
                          0.0, 0.1, config.REWARD_SURVIVAL)
        
        self._create_slider(scrollable_frame, "Reward: Engagement", "REWARD_ENGAGEMENT", 
                          0.0, 0.1, config.REWARD_ENGAGEMENT)
        
        self._create_slider(scrollable_frame, "Reward: Movement Quality", "REWARD_MOVEMENT_QUALITY", 
                          0.0, 0.1, config.REWARD_MOVEMENT_QUALITY)
        
        self._create_slider(scrollable_frame, "Reward: Action Smoothness", "REWARD_ACTION_SMOOTHNESS", 
                          0.0, 0.1, config.REWARD_ACTION_SMOOTHNESS)
        
        self._create_slider(scrollable_frame, "Reward: Momentum", "REWARD_MOMENTUM", 
                          0.0, 0.1, config.REWARD_MOMENTUM)
        
        self._create_slider(scrollable_frame, "Reward: Proximity", "REWARD_PROXIMITY", 
                          0.0, 0.1, config.REWARD_PROXIMITY)
        
        self._create_slider(scrollable_frame, "Reward: Activity", "REWARD_ACTIVITY", 
                          0.0, 0.1, config.REWARD_ACTIVITY)
        
        # Reward clipping
        self._create_slider(scrollable_frame, "Reward Clip Min", "REWARD_CLIP_MIN", 
                          -20.0, 0.0, config.REWARD_CLIP_MIN)
        
        self._create_slider(scrollable_frame, "Reward Clip Max", "REWARD_CLIP_MAX", 
                          0.0, 20.0, config.REWARD_CLIP_MAX)
        
        # Reward Alignment Power
        self._create_slider(scrollable_frame, "Reward Alignment Power", "REWARD_ALIGNMENT_POWER", 
                          1.0, 10.0, config.REWARD_ALIGNMENT_POWER)
        
        # Balance Reward K
        self._create_slider(scrollable_frame, "Balance Reward K", "BALANCE_REWARD_K", 
                          0.0, 1.0, config.BALANCE_REWARD_K)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_input_tab(self, notebook):
        """Create input configuration tab"""
        frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(frame, text="Input")
        
        canvas = tk.Canvas(frame, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1e1e1e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Use Discrete Actions
        self._create_checkbox(scrollable_frame, "Use Discrete Actions", "USE_DISCRETE_ACTIONS")
        
        # Use Physics Controller
        self._create_checkbox(scrollable_frame, "Use Physics Controller", "USE_PHYSICS_CONTROLLER")
        
        # Use Bezier Smoothing
        self._create_checkbox(scrollable_frame, "Use Bezier Smoothing", "USE_BEZIER_SMOOTHING")
        
        # Use DirectInput
        self._create_checkbox(scrollable_frame, "Use DirectInput", "USE_DIRECTINPUT")
        
        # Mouse Sensitivity
        self._create_slider(scrollable_frame, "Mouse Sensitivity", "MOUSE_SENSITIVITY", 
                          10.0, 500.0, config.MOUSE_SENSITIVITY)
        
        # Noise Threshold
        self._create_slider(scrollable_frame, "Noise Threshold", "NOISE_THRESHOLD", 
                          0.1, 5.0, config.NOISE_THRESHOLD)
        
        # Human Timeout
        self._create_slider(scrollable_frame, "Human Timeout (seconds)", "HUMAN_TIMEOUT", 
                          0.1, 2.0, config.HUMAN_TIMEOUT)
        
        # PID Parameters
        self._create_slider(scrollable_frame, "PID Kp", "PID_KP", 
                          0.0, 2.0, config.PID_KP)
        
        self._create_slider(scrollable_frame, "PID Ki", "PID_KI", 
                          0.0, 1.0, config.PID_KI)
        
        self._create_slider(scrollable_frame, "PID Kd", "PID_KD", 
                          0.0, 1.0, config.PID_KD)
        
        self._create_slider(scrollable_frame, "PID Max Output", "PID_MAX_OUTPUT", 
                          0.1, 5.0, config.PID_MAX_OUTPUT)
        
        # Gesture Micro Step Duration
        self._create_slider(scrollable_frame, "Gesture Micro Step Duration", "GESTURE_MICRO_STEP_DURATION", 
                          0.001, 0.1, config.GESTURE_MICRO_STEP_DURATION)
        
        # Frame Skip
        self._create_slider(scrollable_frame, "Frame Skip", "FRAME_SKIP", 
                          1, 10, config.FRAME_SKIP, int_val=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_yolo_tab(self, notebook):
        """Create YOLO configuration tab"""
        frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(frame, text="YOLO")
        
        canvas = tk.Canvas(frame, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1e1e1e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # YOLO Enabled
        self._create_checkbox(scrollable_frame, "YOLO Enabled", "YOLO_ENABLED")
        
        # YOLO Overlay Enabled
        self._create_checkbox(scrollable_frame, "YOLO Overlay Enabled", "YOLO_OVERLAY_ENABLED")
        
        # YOLO Self Learning Enabled
        self._create_checkbox(scrollable_frame, "YOLO Self Learning", "YOLO_SELF_LEARNING_ENABLED")
        
        # YOLO Confidence Adjustment
        self._create_checkbox(scrollable_frame, "YOLO Confidence Adjustment", "YOLO_CONFIDENCE_ADJUSTMENT_ENABLED")
        
        # YOLO Confidence Threshold
        self._create_slider(scrollable_frame, "YOLO Confidence Threshold", "YOLO_CONFIDENCE_THRESHOLD", 
                          0.1, 1.0, config.YOLO_CONFIDENCE_THRESHOLD)
        
        # YOLO Detection Interval
        self._create_slider(scrollable_frame, "YOLO Detection Interval (seconds)", "YOLO_DETECTION_INTERVAL", 
                          0.01, 1.0, config.YOLO_DETECTION_INTERVAL)
        
        # YOLO Min Reward for Labeling
        self._create_slider(scrollable_frame, "YOLO Min Reward for Labeling", "YOLO_MIN_REWARD_FOR_LABELING", 
                          0.0, 2.0, config.YOLO_MIN_REWARD_FOR_LABELING)
        
        # YOLO Self Training Interval
        self._create_slider(scrollable_frame, "YOLO Self Training Interval", "YOLO_SELF_TRAINING_INTERVAL", 
                          100, 10000, config.YOLO_SELF_TRAINING_INTERVAL, int_val=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_performance_tab(self, notebook):
        """Create performance monitoring tab"""
        frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(frame, text="Performance")
        
        # Stats display area
        stats_frame = tk.Frame(frame, bg='#1e1e1e')
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create stats labels
        stats = [
            "FPS", "Frame Count", "Average Reward", "Episode Count",
            "Buffer Size", "Training Loss", "Epsilon", "Model Updates"
        ]
        
        for i, stat in enumerate(stats):
            row = i // 2
            col = (i % 2) * 2
            
            label = tk.Label(stats_frame, text=f"{stat}:", 
                           bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
            label.grid(row=row, column=col, sticky='w', padx=10, pady=5)
            
            value_label = tk.Label(stats_frame, text="N/A", 
                                 bg='#2d2d2d', fg='#00ff00', font=('Arial', 10),
                                 width=20, anchor='w', relief=tk.SUNKEN, padx=5)
            value_label.grid(row=row, column=col+1, sticky='ew', padx=10, pady=5)
            
            self.stats_labels[stat] = value_label
        
        stats_frame.grid_columnconfigure(1, weight=1)
        stats_frame.grid_columnconfigure(3, weight=1)
    
    def _create_control_tab(self, notebook):
        """Create control tab"""
        frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(frame, text="Control")
        
        control_frame = tk.Frame(frame, bg='#1e1e1e')
        control_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Buttons
        button_style = {'bg': '#0078d4', 'fg': 'white', 'font': ('Arial', 12, 'bold'),
                       'width': 20, 'height': 2, 'relief': tk.RAISED, 'bd': 3}
        
        btn_save = tk.Button(control_frame, text="Save Config", command=self.save_config, **button_style)
        btn_save.pack(pady=10)
        
        btn_reset = tk.Button(control_frame, text="Reset to Defaults", command=self.reset_config, **button_style)
        btn_reset.pack(pady=10)
        
        btn_reload = tk.Button(control_frame, text="Reload Config", command=self.reload_config, **button_style)
        btn_reload.pack(pady=10)
        
        # Status
        status_frame = tk.Frame(control_frame, bg='#1e1e1e')
        status_frame.pack(pady=20)
        
        tk.Label(status_frame, text="Status:", bg='#1e1e1e', fg='white', 
                font=('Arial', 12, 'bold')).pack()
        
        self.status_label = tk.Label(status_frame, text="Running", bg='#2d2d2d', 
                                    fg='#00ff00', font=('Arial', 14, 'bold'),
                                    width=30, relief=tk.SUNKEN, padx=10, pady=10)
        self.status_label.pack(pady=10)
    
    def _create_slider(self, parent, label_text, config_key, min_val, max_val, 
                      current_val, int_val=False, log_scale=False):
        """Create a slider widget"""
        frame = tk.Frame(parent, bg='#1e1e1e')
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Label
        label = tk.Label(frame, text=label_text, bg='#1e1e1e', fg='white', 
                        font=('Arial', 9), width=30, anchor='w')
        label.pack(side=tk.LEFT, padx=5)
        
        # Scale
        if log_scale:
            # For log scale, use linear scale but convert values
            scale = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                           bg='#2d2d2d', fg='white', troughcolor='#0078d4',
                           length=300, resolution=0.00001 if not int_val else 1)
        else:
            scale = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                           bg='#2d2d2d', fg='white', troughcolor='#0078d4',
                           length=300, resolution=0.01 if not int_val else 1)
        
        scale.set(current_val)
        scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Value label
        value_label = tk.Label(frame, text=str(current_val), bg='#1e1e1e', 
                             fg='#00ff00', font=('Arial', 9), width=10)
        value_label.pack(side=tk.LEFT, padx=5)
        
        # Update callback
        def update_value(val):
            if int_val:
                val = int(float(val))
            else:
                val = float(val)
            value_label.config(text=f"{val:.4f}" if not int_val else str(val))
            setattr(config, config_key, val)
        
        scale.config(command=update_value)
        
        self.widgets[config_key] = {'scale': scale, 'label': value_label, 'int_val': int_val}
    
    def _create_checkbox(self, parent, label_text, config_key):
        """Create a checkbox widget"""
        frame = tk.Frame(parent, bg='#1e1e1e')
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        var = tk.BooleanVar(value=getattr(config, config_key))
        
        checkbox = tk.Checkbutton(frame, text=label_text, variable=var,
                                 bg='#1e1e1e', fg='white', selectcolor='#0078d4',
                                 activebackground='#2d2d2d', activeforeground='white',
                                 font=('Arial', 9),
                                 command=lambda: setattr(config, config_key, var.get()))
        checkbox.pack(side=tk.LEFT, padx=5)
        
        self.widgets[config_key] = {'var': var, 'checkbox': checkbox}
    
    def _create_dropdown(self, parent, label_text, config_key, options):
        """Create a dropdown widget"""
        frame = tk.Frame(parent, bg='#1e1e1e')
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        label = tk.Label(frame, text=label_text, bg='#1e1e1e', fg='white', 
                        font=('Arial', 9), width=30, anchor='w')
        label.pack(side=tk.LEFT, padx=5)
        
        var = tk.StringVar(value=getattr(config, config_key))
        
        dropdown = ttk.Combobox(frame, textvariable=var, values=options,
                               state='readonly', width=20)
        dropdown.pack(side=tk.LEFT, padx=5)
        
        def update_value(*args):
            setattr(config, config_key, var.get())
        
        var.trace('w', update_value)
        
        self.widgets[config_key] = {'var': var, 'dropdown': dropdown}
    
    def update_yolo_display(self, frame: np.ndarray, detections: Dict):
        """Update YOLO overlay display (called from actor/vision)"""
        with self.yolo_lock:
            if frame is not None:
                self.yolo_frame = frame.copy()
            self.yolo_detections = detections.copy() if detections else {}
    
    def _update_yolo_image(self):
        """Update YOLO display image in GUI"""
        try:
            with self.yolo_lock:
                frame = self.yolo_frame.copy() if self.yolo_frame is not None else None
                detections = self.yolo_detections.copy() if self.yolo_detections else {}
            
            if frame is None or self.yolo_label is None:
                return
            
            # Convert frame to display format
            if len(frame.shape) == 2:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 1:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 3:
                    display_frame = frame.copy()  # Already RGB
                elif frame.shape[2] == 4:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    display_frame = frame.copy()
            else:
                return
            
            # Draw detections
            display_frame = self._draw_detections(display_frame, detections)
            
            # Resize to fit label (larger display area in unified window)
            max_width = 1100
            max_height = 700
            height, width = display_frame.shape[:2]
            scale = min(max_width / width, max_height / height, 1.0)
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
            self.yolo_label.config(image=photo)
            self.yolo_label.image = photo  # Keep a reference
            
        except Exception as e:
            logger.debug(f"Error updating YOLO image: {e}")
    
    def _draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Draw YOLO detections on frame"""
        if not detections or 'objects' not in detections:
            return frame
        
        display_frame = frame.copy()
        class_colors = {
            'Blood': (255, 0, 0),      # Red
            'Enemy': (0, 0, 255),      # Blue
            'Player': (0, 255, 0),     # Green
            'You Won': (255, 255, 0), # Yellow
        }
        
        for obj in detections.get('objects', []):
            bbox = obj.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            class_name = obj.get('class_name', 'Unknown')
            confidence = obj.get('confidence', 0.0)
            
            # Get color
            color = class_colors.get(class_name, (255, 165, 0))
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return display_frame
    
    def _update_loop(self):
        """Update loop for real-time stats and YOLO display"""
        while self.running and self.root:
            try:
                if not self.stats_labels:
                    time.sleep(0.5)
                    continue
                    
                if self.performance_monitor:
                    with self.performance_monitor.lock:
                        fps = self.performance_monitor.metrics.get('fps', [0])
                        fps_val = fps[-1] if fps else 0
                        frame_count = self.performance_monitor.frame_count
                        avg_reward = self.performance_monitor.metrics.get('reward', [0])
                        avg_reward_val = sum(avg_reward[-100:]) / len(avg_reward[-100:]) if avg_reward else 0
                    
                    if 'FPS' in self.stats_labels:
                        self.root.after(0, lambda v=fps_val: self.stats_labels['FPS'].config(text=f"{v:.1f}"))
                    if 'Frame Count' in self.stats_labels:
                        self.root.after(0, lambda v=frame_count: self.stats_labels['Frame Count'].config(text=str(v)))
                    if 'Average Reward' in self.stats_labels:
                        self.root.after(0, lambda v=avg_reward_val: self.stats_labels['Average Reward'].config(text=f"{v:.3f}"))
                
                if self.learner and 'Buffer Size' in self.stats_labels:
                    buffer_size = len(self.learner.replay_buffer) if hasattr(self.learner, 'replay_buffer') else 0
                    self.root.after(0, lambda v=buffer_size: self.stats_labels['Buffer Size'].config(text=str(v)))
                
                if self.actor and 'Epsilon' in self.stats_labels:
                    epsilon = getattr(self.actor, 'epsilon', 0) if hasattr(self.actor, 'epsilon') else config.DQN_EPSILON_START
                    self.root.after(0, lambda v=epsilon: self.stats_labels['Epsilon'].config(text=f"{v:.3f}"))
                
                # Update YOLO display
                self.root.after(0, self._update_yolo_image)
                
            except Exception as e:
                logger.error(f"Error updating stats: {e}")
            
            time.sleep(0.033)  # Update ~30 FPS for smooth YOLO display
    
    def save_config(self):
        """Save current config"""
        messagebox.showinfo("Config Saved", "Configuration has been updated in memory.\nNote: Changes are not persisted to disk.")
    
    def reset_config(self):
        """Reset config to original values"""
        if messagebox.askyesno("Reset Config", "Reset all settings to defaults?"):
            for key, value in self.original_config.items():
                try:
                    setattr(config, key, value)
                    if key in self.widgets:
                        widget = self.widgets[key]
                        if 'scale' in widget:
                            widget['scale'].set(value)
                        elif 'var' in widget:
                            widget['var'].set(value)
                except:
                    pass
            messagebox.showinfo("Config Reset", "Configuration reset to defaults.")
    
    def reload_config(self):
        """Reload config from current values"""
        for key in self.widgets:
            try:
                current_val = getattr(config, key)
                widget = self.widgets[key]
                if 'scale' in widget:
                    widget['scale'].set(current_val)
                elif 'var' in widget:
                    widget['var'].set(current_val)
            except:
                pass
        messagebox.showinfo("Config Reloaded", "Configuration reloaded from current values.")
    
    def on_closing(self):
        """Handle window closing - properly shutdown everything"""
        logger.info("Dashboard window closing - shutting down...")
        self.running = False
        
        # Stop agent if available
        if self.agent:
            try:
                self.agent.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent: {e}")
        
        # Destroy window
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
        
        # Force exit if needed
        import sys
        import os
        logger.info("Dashboard closed - exiting program")
        os._exit(0)  # Force exit to ensure all threads stop

