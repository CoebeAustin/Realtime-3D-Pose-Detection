import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

tf.get_logger().setLevel('ERROR')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize the Pose model.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks_without_face(image, landmarks):
    """
    Draw pose landmarks on an image, excluding face landmarks.
    Args:
        image: The image to draw on.
        landmarks: The list of landmarks to draw.
    """
    # Define connections that do not involve face landmarks.
    pose_connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    ]

    # Draw the landmarks and connections.
    for idx, landmark in enumerate(landmarks.landmark):
        if idx > 10:  # Skip face landmarks.
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            z = landmark.z
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    for connection in pose_connections:
        start_idx = connection[0].value
        end_idx = connection[1].value
        if start_idx > 10 and end_idx > 10:
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]
            start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
            end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

def detectPose(image, pose):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks need to be detected.
        pose: The pose setup function required to perform the pose detection.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:
        # Draw Pose landmarks on the output image, excluding face landmarks.
        draw_landmarks_without_face(output_image, results.pose_landmarks)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))
    else:
        # If no landmarks are detected, display a message on the image.
        cv2.putText(output_image, "No bodies detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Return the output image and the found landmarks.
    return output_image, landmarks

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initialize the webcam.
cap = cv2.VideoCapture(0)
counter = 0
stage  = None

# Check if the webcam is opened correctly.
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the frame width and height (optional).
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    # Read a frame from the webcam.
    ret, frame = cap.read()

    # If the frame is read correctly, ret will be True.
    if not ret:
        break

    # Perform pose detection.
    output_frame, landmarks = detectPose(frame, pose)

    if landmarks:
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]]
        angle = calculate_angle(shoulder, elbow, wrist)
        cv2.putText(output_frame, str(angle), tuple(elbow), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if angle > 160:
            stage = "down"
        if angle < 30 and stage == "down":
            stage = "up"
            counter += 1
            print(counter)
            
    # Display the output frame.
    cv2.imshow('Pose Detection', output_frame)

    # Break the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window.
cap.release()
cv2.destroyAllWindows()
