import cv2
import numpy as np
import pygame
import time  # To track time between plays

# Initialize pygame mixer
pygame.mixer.init()

# Load your sound
sound = pygame.mixer.Sound(r"C:\Users\BERNARD\Documents\Basic OpenCV Project\Dramatic Vine_Instagram Boom - Sound Effect (HD).mp3")

# Load the model and classes
net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\BERNARD\Documents\Basic OpenCV Project\deploy.prototxt",
    r"C:\Users\BERNARD\Documents\Basic OpenCV Project\mobilenet_iter_73000.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", 
           "sofa", "train", "tvmonitor"]

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

last_played_time = 0  # Track last time sound was played
cooldown_seconds = 5  # Time to wait before next sound

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    current_time = time.time()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if label == "person":
                color = (0, 0, 255)  # Red

                # Check cooldown before playing sound
                if current_time - last_played_time >= cooldown_seconds:
                    print("Person detected! Boom sound playing.")
                    sound.play()
                    last_played_time = current_time

            else:
                color = (0, 255, 0)  # Green

            label_text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
