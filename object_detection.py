import cv2
import torch
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 1 for an external webcam

if not cap.isOpened():
    print("❌ Error: Could not access the webcam")
    exit()

plt.ion()  # Enable interactive mode for real-time updates

try:
    while True:
        start_time = time.time()  # Track frame processing time

        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes
        frame = results[0].plot()

        # Convert BGR to RGB (for Matplotlib)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame using Matplotlib
        plt.imshow(frame_rgb)
        plt.axis("off")
        plt.draw()
        plt.pause(0.01)  # Pause for real-time update
        plt.clf()  # Clear previous frame

        # Ensure each frame is displayed for at least 1 second
        time.sleep(max(1 - (time.time() - start_time), 0))

except KeyboardInterrupt:
    print("\n🛑 Stopping detection...")

finally:
    cap.release()
    plt.close()
    print("✅ Webcam Closed Successfully")
