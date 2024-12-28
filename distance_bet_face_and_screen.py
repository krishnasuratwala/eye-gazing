import cv2
import mediapipe as mp

# Constants (update with the values from Script 1)
real_world_eye_distance = 8.5  # Distance between eyes in cm
focal_length = 254.12 # Replace with your calculated focal length from Script 1

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5)

# Open the camera
cap = cv2.VideoCapture(0)

print("Calculating distance. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image color to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get left and right eye landmarks
            left_eye = 473
            right_eye = 468

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            left_eye_x, left_eye_y = int(face_landmarks.landmark[left_eye].x * w), int(face_landmarks.landmark[left_eye].y * h)
            right_eye_x, right_eye_y = int(face_landmarks.landmark[right_eye].x * w), int(face_landmarks.landmark[right_eye].y * h)
            
            # Calculate pixel distance between left and right eye
            pixel_eye_distance = ((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2) ** 0.5
            
            # Calculate distance from camera
            distance = (real_world_eye_distance * focal_length) / pixel_eye_distance
            print(f"Distance from camera: {distance:.2f} cm")

    # Display the frame with landmarks
    cv2.imshow("Distance Measurement", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
