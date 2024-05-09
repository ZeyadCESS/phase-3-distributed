import tkinter as tk
from tkinter import filedialog
import threading
import queue
import cv2
import numpy as np

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        super().__init__()
        self.task_queue = task_queue

    def run(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            image, operation, filter_type, rotation_angle, resize_scale = task
            result = self.process_image(image, operation, filter_type, rotation_angle, resize_scale)
            self.display_result(result)

    def process_image(self, image, operation, filter_type, rotation_angle, resize_scale):
        img = cv2.imread(image, cv2.IMREAD_COLOR)

        # Apply specified operations
        if operation == 'edge_detection':
            result = cv2.Canny(img, 100, 200)
        elif operation == 'color_inversion':
            result = cv2.bitwise_not(img)
        else:
            result = img  # Default to original image if no specific operation is selected

        # Apply rotation
        if rotation_angle != 0:
            rows, cols = result.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
            result = cv2.warpAffine(result, rotation_matrix, (cols, rows))

        # Apply resizing
        if resize_scale != 1.0:
            result = cv2.resize(result, None, fx=resize_scale, fy=resize_scale)

        # Apply the selected filter
        if filter_type == 'grayscale':
            if len(result.shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        elif filter_type == 'blur':
            result = cv2.GaussianBlur(result, (5, 5), 0)
        elif filter_type == 'custom':
            # Define a custom filter kernel
            custom_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            # Apply the custom filter
            result = cv2.filter2D(result, -1, custom_filter)
        # Add more filters as needed...
        return result

    def display_result(self, result):
        cv2.imshow('Processed Image', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def select_image():
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def process_image():
    image_path = entry.get()
    operation = var.get()
    filter_type = filter_var.get()
    rotation_angle = rotation_scale.get()
    resize_scale = resize_scale_var.get()
    task_queue.put((image_path, operation, filter_type, rotation_angle, resize_scale))
    start_worker_thread()

def start_worker_thread():
    WorkerThread(task_queue).start()

root = tk.Tk()
root.title("Advanced Image Processing")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

label1 = tk.Label(frame, text="Select Image:")
label1.grid(row=0, column=0, sticky="w")

entry = tk.Entry(frame, width=50)
entry.grid(row=0, column=1, padx=5, pady=5)

button = tk.Button(frame, text="Browse", command=select_image)
button.grid(row=0, column=2, padx=5, pady=5)

label2 = tk.Label(frame, text="Select Operation:")
label2.grid(row=1, column=0, sticky="w")

var = tk.StringVar()
var.set("edge_detection")

option1 = tk.Radiobutton(frame, text="Edge Detection", variable=var, value="edge_detection")
option1.grid(row=1, column=1, sticky="w")

option2 = tk.Radiobutton(frame, text="Color Inversion", variable=var, value="color_inversion")
option2.grid(row=2, column=1, sticky="w")

label3 = tk.Label(frame, text="Select Filter:")
label3.grid(row=3, column=0, sticky="w")

filter_var = tk.StringVar()
filter_var.set("grayscale")

filter_option1 = tk.Radiobutton(frame, text="Grayscale", variable=filter_var, value="grayscale")
filter_option1.grid(row=3, column=1, sticky="w")

filter_option2 = tk.Radiobutton(frame, text="Blur", variable=filter_var, value="blur")
filter_option2.grid(row=4, column=1, sticky="w")

filter_option3 = tk.Radiobutton(frame, text="Custom Filter", variable=filter_var, value="custom")
filter_option3.grid(row=5, column=1, sticky="w")

label4 = tk.Label(frame, text="Rotation Angle (degrees):")
label4.grid(row=6, column=0, sticky="w")

rotation_scale = tk.Scale(frame, from_=0, to=360, orient=tk.HORIZONTAL)
rotation_scale.grid(row=6, column=1, padx=5, pady=5)

label5 = tk.Label(frame, text="Resize Scale:")
label5.grid(row=7, column=0, sticky="w")

resize_scale_var = tk.DoubleVar()
resize_entry = tk.Entry(frame, textvariable=resize_scale_var, width=10)
resize_entry.grid(row=7, column=1, padx=5, pady=5)
resize_scale_var.set(1.0)

process_button = tk.Button(frame, text="Process Image", command=process_image)
process_button.grid(row=8, column=1, padx=5, pady=10)

task_queue = queue.Queue()

root.mainloop()
