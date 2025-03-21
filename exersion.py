import cv2
import mediapipe as mp
import numpy as np

# Initialize the Pose model.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks_without_face(image, landmarks):
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
        # If no landmarks are detected, display a centered message on the image.
        text = "No bodies detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (0, 0, 255)

        # Get the size of the text box
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Calculate the position for centered text
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        # Put the text on the image
        cv2.putText(output_image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Return the output image and the found landmarks.
    return output_image, landmarks

def calculate_angle(x, y, z):
    a = np.array(x)
    b = np.array(y)
    c = np.array(z)

    radians = np.arctan2(z[1]-y[1], z[0]-b[0]) - np.arctan2(x[1] - y[1], x[0] - y[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to track curls
def track_curls(landmarks):
    global left_counter, right_counter, left_stage, right_stage

    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][0], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][1]]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]]

    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    cv2.putText(output_frame, str(left_angle), tuple(left_elbow), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_frame, str(right_angle), tuple(right_elbow), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    if left_angle > 160:
        left_stage = "down"
    if left_angle < 30 and left_stage == "down":
        left_stage = "up"
        left_counter += 1
        print(f"Left counter: {left_counter}")

    if right_angle > 160:
        right_stage = "down"
    if right_angle < 30 and right_stage == "down":
        right_stage = "up"
        right_counter += 1
        print(f"Right counter: {right_counter}")

# Function to track squats
def track_squats(landmarks):
    global left_counter, right_counter, left_stage, right_stage

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][0], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][1]]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][1]]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][0], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][1]]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1]]

    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    cv2.putText(output_frame, str(left_angle), tuple(left_knee), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_frame, str(right_angle), tuple(right_knee), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    if left_angle and right_angle > 160:
        left_stage = "up"
    if left_angle and right_angle < 90 and left_stage == "up":
        left_stage = "down"
        left_counter += 1
        print(f"Squat counter: {left_counter}")

# Initialize the webcam.
cap = cv2.VideoCapture(0)
left_counter = 0
right_counter = 0
left_stage = None
right_stage = None

# Check if the webcam is opened correctly.
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the frame width and height (optional).
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get user input for exercise type
exercise = input("Enter the exercise to track. Please enter \'s\' for squats, enter \'c\' for curls. Please enter \'q\' if you want to terminate or you are finished using the app: ").strip().lower()

while cap.isOpened():
    # Read a frame from the webcam.
    ret, frame = cap.read()

    # If the frame is read correctly, ret will be True.
    if not ret:
        break

    # Perform pose detection.
    output_frame, landmarks = detectPose(frame, pose)

    if landmarks:
        if exercise == "c":
            track_curls(landmarks)
            # Display the left counter on the top-left side
            cv2.putText(output_frame, f'Left Counter: {left_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Display the right counter on the top-right side
            right_text = f'Right Counter: {right_counter}'
            text_size = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(output_frame, right_text, (frame.shape[1] - text_size[0] - 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Display the output frame.
            cv2.imshow('Pose Detection', output_frame)
        elif exercise == "s":
            track_squats(landmarks)
            # Display the left counter on the top-left side
            cv2.putText(output_frame, f'Squat Counter: {left_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Display the output frame.
            cv2.imshow('Pose Detection', output_frame)

        elif exercise == "q":
            quit()

        else:
            print("Invalid exercise type. Please restart the program and enter 'squats' or 'curls'.")
            break


    # Break the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window.
cap.release()
cv2.destroyAllWindows()
