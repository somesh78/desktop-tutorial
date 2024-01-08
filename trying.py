import cv2
import numpy as np
import tensorflow as tf
import threading

# Load the model
model_path = r"C:\Users\DELL\PythonProjects\object identifying algos\trying\efficientdet_d4_coco17_tpu-32\efficientdet_d4_coco17_tpu-32\saved_model"  # Replace with your actual path

try:
    model = tf.saved_model.load(model_path)
except Exception as e:
    print("Error loading the model:", e)
    exit(1)

# Create a dictionary mapping class indices to class names
class_names = {
    1: 'Person',
    2: 'Car',
    3: 'Bicycle',
    4: 'Motorcycle',
    5: 'Bus',
    6: 'Truck',
    7: 'Traffic Light',
    8: 'Stop Sign',
    9: 'Pedestrian',
    10: 'Dog',
    11: 'Cat',
    12: 'Bird',
    13: 'Chair',
    14: 'Table',
    15: 'Couch',
    16: 'Bed',
    17: 'Laptop',
    18: 'Monitor',
    19: 'Keyboard',
    20: 'Mouse',
    21: 'Smartphone',
    22: 'Book',
    23: 'Bottle',
    24: 'Cup',
    25: 'Plate',
    26: 'Fork',
    27: 'Knife',
    28: 'Spoon',
    29: 'Bowl',
    30: 'Plant',
    # Add more classes as needed or based on your model's output
}

# Set up the camera
width, height = 640, 480  # Adjust the dimensions as needed
cap = cv2.VideoCapture(0)  # Change the value to use a different camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit(1)

def inference_worker():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to receive frame from the camera.")
            break

        # Preprocess the input frame
        try:
            resized_frame = cv2.resize(frame, (512, 512))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.uint8)
            input_tensor = tf.expand_dims(input_tensor, 0)
        except Exception as e:
            print("Error preprocessing frame:", e)
            break

        try:
            detections = model(input_tensor)

            # Placeholder: Extracting bounding boxes, classes, and scores
            boxes = detections['detection_boxes'][0].numpy()  # Update this line
            classes = detections['detection_classes'][0].numpy().astype(np.int32)  # Update this line
            scores = detections['detection_scores'][0].numpy()  # Update this line

            display_frame = frame.copy()  # Create a copy of the frame for displaying results

            h, w, _ = frame.shape
            for i in range(len(boxes)):
                if scores[i] > 0.5:  # Change the threshold as needed
                    ymin, xmin, ymax, xmax = boxes[i]
                    ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)

                    class_index = classes[i]
                    if class_index in class_names:
                        label = class_names[class_index]
                    else:
                        label = f'Class {class_index}'
                    score = scores[i]

                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(display_frame, f'{label}: {score:.2f}', (xmin, ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Object Detection', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("Error during inference:", e)
            break

    cap.release()
    cv2.destroyAllWindows()

# Run inference on a separate thread
inference_thread = threading.Thread(target=inference_worker)
inference_thread.start()
