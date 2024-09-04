import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import find_peaks
import threading

def GetThreshold(input_video_path, distance=30, prominence=1000):
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize histogram accumulation
    hist_accum = np.zeros((256,), dtype=np.float32)

    for _ in range(frame_count):
        res, frame = cap.read()
        if not res:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        
        hist_accum += hist.flatten()

    # Calculate average histogram
    hist_avg = hist_accum / frame_count
    
    # Replace zero values with the average of neighboring bins
    for i in range(1, len(hist_avg) - 1):
        if hist_avg[i] == 0:
            hist_avg[i] = (hist_avg[i - 1] + hist_avg[i + 1]) / 2
    
    # Handle the first and last bins if they are zero
    if hist_avg[0] == 0:
        hist_avg[0] = hist_avg[1]
    if hist_avg[-1] == 0:
        hist_avg[-1] = hist_avg[-2]

    # Find local minima
    peaks, _ = find_peaks(-hist_avg, distance=distance, prominence=prominence)

    return peaks[0]

def VideoModification(input_video_path, output_video_path, threshold):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    frame_width =   int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps =           int(cap.get(cv2.CAP_PROP_FPS))
    frame_count =   int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the codec and create VideoWriter object
    fourcc =    cv2.VideoWriter_fourcc(*'mp4v')
    out =       cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    for frame_num in range(frame_count):
        res, frame = cap.read()
        if not res:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Copy the grayscale frame
        gray_frame_copy = gray_frame.copy()
        
        # Erase the background and any other bright object
        gray_frame_copy[gray_frame > threshold + 2] = 0
        gray_frame_copy[gray_frame < threshold - 2] = 0
        
        # Stretch the pixel values
        gray_frame_copy[gray_frame_copy != 0] = 255
        
        # Add the mask to the original frame
        gray_frame[gray_frame_copy == 255] = 255
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Write the frame into the output video
        out.write(gray_frame)
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print('Video Creation Completed.')

class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor")
        
        self.input_video_paths = []
        self.output_video_directory = tk.StringVar()
        
        tk.Label(root, text="Input Video Paths:").grid(row=0, column=0, sticky='e', padx=10, pady=10)
        self.input_videos_entry = tk.Entry(root, width=50)
        self.input_videos_entry.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(root, text="Browse", command=self.browse_input_videos).grid(row=0, column=2, padx=10, pady=10)
        
        tk.Label(root, text="Output Directory:").grid(row=1, column=0, sticky='e', padx=10, pady=10)
        tk.Entry(root, textvariable=self.output_video_directory, width=50).grid(row=1, column=1, padx=10, pady=10)
        tk.Button(root, text="Browse", command=self.browse_output_directory).grid(row=1, column=2, padx=10, pady=10)
        
        tk.Button(root, text="Process Videos", command=self.process_videos).grid(row=2, column=1, pady=10, padx=(0, 150))
        tk.Button(root, text="Close", command=root.quit).grid(row=2, column=1, pady=10)


    
    def browse_input_videos(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4;*.avi"), ("All files", "*.*")])
        if file_paths:
            self.input_video_paths = list(file_paths)
            self.input_videos_entry.delete(0, tk.END)
            self.input_videos_entry.insert(0, '; '.join(file_paths))
    
    def browse_output_directory(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.output_video_directory.set(directory_path)
    
    def process_videos(self):
        output_directory = self.output_video_directory.get()
        
        if not self.input_video_paths or not output_directory:
            messagebox.showerror("Error", "Please specify both input video paths and output directory.")
            return
        
        # Run the video processing in a separate thread to avoid freezing the GUI
        threading.Thread(target=self.run_video_processing).start()
    
    def run_video_processing(self):
        try:
            for input_video_path in self.input_video_paths:
                video_filename = input_video_path.split("/")[-1]
                output_video_path = f"{self.output_video_directory.get()}/preprocessed_{video_filename.split('.')[0]}.mp4"
                
                threshold = GetThreshold(input_video_path)
                VideoModification(input_video_path, output_video_path, threshold)
            
            messagebox.showinfo("Success", "Video processing completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()