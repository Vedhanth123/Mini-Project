import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import pyttsx3

# Initialize the YOLOv8 model (nano model for better performance)
model = YOLO('yolov8n.pt')

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize pyttsx3
engine = pyttsx3.init()

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"Failed to play sound: {e}")

def detect_objects(frame):
    results = model(frame)
    detected_objects = []

    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates
        scores = result.boxes.conf.numpy()  # Confidence scores
        class_ids = result.boxes.cls.numpy()  # Class IDs

        for i in range(len(boxes)):
            if scores[i] > 0.5:
                x1, y1, x2, y2 = boxes[i]
                label = result.names[int(class_ids[i])]
                detected_objects.append((label, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, detected_objects

def perform_ocr(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = frame[y1:y2, x1:x2]
    text = pytesseract.image_to_string(roi)
    return text

try:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame_count += 1

        # Only process every 5th frame to reduce load
        if frame_count % 5 != 0:
            continue

        # Resize frame to reduce computation
        frame = cv2.resize(frame, (320, 240))

        frame, detected_objects = detect_objects(frame)

        key = cv2.waitKey(1) & 0xFF

        if detected_objects:
            for obj, bbox in detected_objects:
                text = perform_ocr(frame, bbox)
                if text.strip():
                    speak(f"Detected {obj} and text: {text.strip()}")

        cv2.imshow("Frame", frame)
        if key == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
