import cv2
import numpy as np
import tensorflow as tf
import os
import time

# Load YOLOv4 model and configuration
net = cv2.dnn.readNet(r'C:\Users\02avi\OneDrive\Desktop\Hackathon\yolov4-tiny.weights', r'C:\Users\02avi\OneDrive\Desktop\Hackathon\yolov4-tiny.cfg')

# Load COCO class names
classes = []
with open(r'C:\Users\02avi\OneDrive\Desktop\Hackathon\coco.names', 'r') as f:
    classes = f.read().splitlines()

# Load the accident detection model
accident_model = tf.keras.models.load_model(r'C:\Users\02avi\OneDrive\Desktop\Hackathon\accident.h5')

# Open the video file
video_path = r'C:\Users\02avi\OneDrive\Desktop\Hackathon\1.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the time per frame
time_per_frame = 20 / fps

# Conversion factor from pixels/second to meters/second
pixel_to_meter = 0.0277

# Initialize variables for vehicle tracking
previous_frame = None

# Initialize variables for video clip recording
is_first_collision_detected = False
accident_start_time = None
video_writer = None
output_file_counter = 1

# Define minimum bounding box area to consider for speed calculation
min_box_area = 1000  # Adjust this value according to your requirement

# Initialize list to store speeds of colliding vehicles
colliding_vehicle_speeds = []

# Assign the number of cars involved in the crash
cars_involved_count = 2

# Define the output directory for text files
output_directory = r'C:\Users\02avi\OneDrive\Desktop\Hackathon\TXT Files'

# Function to generate collision data file name
def generate_collision_data_file_name(counter):
    return os.path.join(output_directory, f'collision_data_{counter}.txt')

# Function to write collision data to text file
def write_collision_data(speed1, speed2, counter, num_cars):
    file_path = generate_collision_data_file_name(counter)
    with open(file_path, 'w') as f:
        f.write("Speed of colliding vehicles just before collision:\n")
        f.write(f"Vehicle 1: {speed1:.2f} meters/second\n")
        f.write(f"Vehicle 2: {speed2:.2f} meters/second\n")
        f.write(f"Number of cars involved: {num_cars}\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class ID 2 corresponds to "car" in COCO dataset
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add bounding box, confidence, and class ID to lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Process the remaining detections after NMS
    if len(indices) > 0:
        for idx in indices.flatten():
            box = boxes[idx]
            x, y, w, h = box

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate speed and overlay speed information
            box_area = w * h
            if box_area > min_box_area:
                if previous_frame is not None:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    displacement = (center_x - previous_frame[0], center_y - previous_frame[1])
                    distance_pixels = np.linalg.norm(displacement)
                    speed_pixels_per_second = distance_pixels / time_per_frame
                    speed_meters_per_second = speed_pixels_per_second * pixel_to_meter
                    cv2.putText(frame, f'Speed: {speed_meters_per_second:.2f} meters/second', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Append speed to colliding_vehicle_speeds list
                    colliding_vehicle_speeds.append(speed_meters_per_second)

                # Update previous frame
                previous_frame = (x + w // 2, y + h // 2)

    # Preprocess the frame for accident detection
    img_width = 250
    img_height = 250
    frame_accident = cv2.resize(frame, (img_width, img_height))
    frame_accident = frame_accident / 255.0
    frame_accident = np.expand_dims(frame_accident, axis=0)

    # Run the accident detection model
    prediction = accident_model.predict(frame_accident)

    # Draw "Accident Detected!" text if an accident is predicted
    if np.argmax(prediction) == 1:  # Assuming class 1 is "accident"
        if not is_first_collision_detected:
            is_first_collision_detected = True
            accident_start_time = time.time()

            # Draw "Accident Detected!" text on the frame
            cv2.putText(frame, "Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Print speeds of colliding vehicles just before the collision
            if len(colliding_vehicle_speeds) >= 2:
                speed1 = colliding_vehicle_speeds[-2]
                speed2 = colliding_vehicle_speeds[-1]
                print("Speed of colliding vehicles just before collision:")
                print(f"Vehicle 1: {speed1:.2f} meters/second")
                print(f"Vehicle 2: {speed2:.2f} meters/second")

                # Write collision data to text file
                write_collision_data(speed1, speed2, output_file_counter, cars_involved_count)

            # If video_writer is not created yet, create it
            if video_writer is None:
                # Define the output file name for the video clip
                output_file_name = os.path.join(r'C:\Users\02avi\OneDrive\Desktop\Hackathon\Recording', f'accident_clip_{output_file_counter}.mp4')

                # Create the video writer
                video_writer = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1], frame.shape[0]))

    # Write frame to the video clip if the first collision is detected
    if is_first_collision_detected:
        video_writer.write(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer
if video_writer is not None:
    video_writer.release()

# Release the video capture
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()