
import cv2
import numpy as np
import tensorflow as tf
import time

# Load YOLOv4 model and configuration
net = cv2.dnn.readNet(r'/home/vansh/Documents/csa_hackathon/yolov4-tiny.weights', r'/home/vansh/Documents/csa_hackathon/yolov4-tiny.cfg')

# Load COCO class names
classes = []
with open(r'/home/vansh/Documents/csa_hackathon/yolov4/coco.names', 'r') as f:
    classes = f.read().splitlines()

# Load the accident detection model
accident_model = tf.keras.models.load_model(r'accident.h5')

# Open the video file
video_path = r'/home/vansh/Documents/csa_hackathon/test-videos/testing3.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize variables for vehicle tracking
previous_frame = None

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the time per frame
time_per_frame = 20 / fps

# Conversion factor from pixels/second to meters/second
pixel_to_meter = 0.0277

# Define minimum bounding box area to consider for speed calculation
min_box_area = 1000 # Adjust this value according to your requirement

# Confidence threshold for accident detection
confidence_threshold = 0.90

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2: # Class ID 2 corresponds to "car" in COCO dataset
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate speed and overlay speed information
                box_area = w * h
                if box_area > min_box_area:
                    if previous_frame is not None:
                        displacement = (center_x - previous_frame[0], center_y - previous_frame[1])
                        distance_pixels = np.linalg.norm(displacement)
                        speed_pixels_per_second = distance_pixels / time_per_frame
                        speed_meters_per_second = speed_pixels_per_second * pixel_to_meter
                        cv2.putText(frame, f'Speed: {speed_meters_per_second:.2f} meters/second', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Update previous frame
                    previous_frame = (center_x, center_y)

    # Preprocess the frame for accident detection
    img_width = 250
    img_height = 250
    frame_accident = cv2.resize(frame, (img_width, img_height))
    frame_accident = frame_accident / 255.0
    frame_accident = np.expand_dims(frame_accident, axis=0)

    # Run the accident detection model
    prediction = accident_model.predict(frame_accident)

    # Draw "Accident Detected!" text if an accident is predicted with high confidence
    if np.argmax(prediction) == 1 and np.max(prediction) > confidence_threshold:
        cv2.putText(frame, "Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()