import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def percent_stretch(img, lower_percent=2, upper_percent=98):
    if len(img.shape) == 3:  # Color image
        # Process each channel separately
        channels = cv2.split(img)
        stretched_channels = []
        
        for channel in channels:
            lower_val = np.percentile(channel, lower_percent)
            upper_val = np.percentile(channel, upper_percent)
            
            stretched = (channel - lower_val) * (255 / (upper_val - lower_val))
            stretched = np.clip(stretched, 0, 255)
            stretched = stretched.astype(np.uint8)
            stretched_channels.append(stretched)
        
        # Merge the channels back together
        stretched = cv2.merge(stretched_channels)
    else:  # Grayscale image
        lower_val = np.percentile(img, lower_percent)
        upper_val = np.percentile(img, upper_percent)

        stretched = (img - lower_val) * (255 / (upper_val - lower_val))
        stretched = np.clip(stretched, 0, 255)
        stretched = stretched.astype(np.uint8)

    return stretched

class ImageEnhancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Enhancer")
        self.root.geometry("1200x700")
        
        # Variables
        self.folder_path = ""
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.current_image_rgb = None  # Store RGB version for display
        self.enhanced_image = None
        self.processing_time = 0
        self.enhanced = False  # Track if image has been enhanced
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Folder selection frame
        folder_frame = tk.Frame(self.root)
        folder_frame.pack(pady=10)
        
        tk.Button(folder_frame, text="Select Folder", command=self.select_folder).pack(side=tk.LEFT)
        self.folder_label = tk.Label(folder_frame, text="No folder selected")
        self.folder_label.pack(side=tk.LEFT, padx=10)
        
        # Image display frame - two canvases side by side
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Left canvas for original image
        left_frame = tk.Frame(self.image_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.left_label = tk.Label(left_frame, text="Original Image")
        self.left_label.pack()
        self.canvas_original = tk.Canvas(left_frame, bg="gray")
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        
        # Right canvas for enhanced image
        right_frame = tk.Frame(self.image_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.right_label = tk.Label(right_frame, text="Enhanced Image")
        self.right_label.pack()
        self.canvas_enhanced = tk.Canvas(right_frame, bg="gray")
        self.canvas_enhanced.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(pady=10)
        
        tk.Button(controls_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Enhance", command=self.enhance_image).pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=5)
        
        self.status_label = tk.Label(status_frame, text="Ready")
        self.status_label.pack()
        
        self.time_label = tk.Label(status_frame, text="Processing time: 0.00s")
        self.time_label.pack()
        
    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path = folder_selected
            self.folder_label.config(text=f"Selected: {os.path.basename(folder_selected)}")
            self.load_images()
            
    def load_images(self):
        if not self.folder_path:
            return
            
        # Get all image files from folder
        self.image_files = []
        for file in os.listdir(self.folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                self.image_files.append(file)
                
        if not self.image_files:
            messagebox.showwarning("No Images", "No image files found in the selected folder")
            return
            
        self.current_index = 0
        self.enhanced = False
        self.show_image()
        
    def show_image(self):
        if not self.image_files:
            return
            
        # Load and display image
        image_path = os.path.join(self.folder_path, self.image_files[self.current_index])
        self.current_image = cv2.imread(image_path)  # Load in color mode
        
        if self.current_image is None:
            messagebox.showerror("Error", f"Cannot load image: {self.image_files[self.current_index]}")
            return
            
        # Update status
        self.status_label.config(text=f"{self.image_files[self.current_index]} ({self.current_index+1}/{len(self.image_files)})")
        
        # Convert BGR to RGB for display
        self.current_image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Display original image on left canvas
        self.display_image(self.current_image_rgb, self.canvas_original)
        
        # Clear enhanced image on right canvas if not enhanced yet
        if not self.enhanced:
            self.canvas_enhanced.delete("all")
            self.canvas_enhanced.create_text(
                self.canvas_enhanced.winfo_width()//2, 
                self.canvas_enhanced.winfo_height()//2, 
                text="Click 'Enhance' to see result", 
                fill="white", 
                font=("Arial", 14)
            )
        
    def display_image(self, img, canvas):
        # Convert to PIL Image
        pil_image = Image.fromarray(img)
        
        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Use default size if canvas not yet sized
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 500, 400
            
        img_width, img_height = pil_image.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display on canvas
        canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        canvas.image = photo  # Keep a reference
        
    def previous_image(self):
        if not self.image_files:
            return
            
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.enhanced = False
        self.enhanced_image = None
        self.time_label.config(text="Processing time: 0.00s")
        self.show_image()
        
    def next_image(self):
        if not self.image_files:
            return
            
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.enhanced = False
        self.enhanced_image = None
        self.time_label.config(text="Processing time: 0.00s")
        self.show_image()
        
    def enhance_image(self):
        if self.current_image is None:
            return
            
        # Process image and time it
        start_time = time.time()
        enhanced_bgr = percent_stretch(self.current_image, 2, 98)  # Work on BGR image
        os.makedirs("enhanced", exist_ok=True)
        cv2.imwrite("./enhanced/enhanced.jpg", enhanced_bgr)
        self.enhanced_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
        end_time = time.time()
        
        self.processing_time = end_time - start_time
        self.time_label.config(text=f"Processing time: {self.processing_time:.2f}s")
        
        # Display enhanced image on right canvas
        self.display_image(self.enhanced_image, self.canvas_enhanced)
        self.enhanced = True

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEnhancerApp(root)
    root.mainloop()