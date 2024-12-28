import cv2
import mediapipe as mp

# Known values
real_world_eye_distance = 8.5  # Distance between eyes in cm (measure your own for accuracy)
known_distance_from_camera = 45  # Known distance from face to camera in cm

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5)

# Open the camera
cap = cv2.VideoCapture(0)

print("Please position your face at the known distance (30 cm). Press 'q' to capture.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image color to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get left and right eye landmarks
            left_eye_index = 473 
            right_eye_index = 468

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            left_eye_x, left_eye_y = int(face_landmarks.landmark[left_eye_index].x * w), int(face_landmarks.landmark[left_eye_index].y * h)
            right_eye_x, right_eye_y = int(face_landmarks.landmark[right_eye_index].x * w), int(face_landmarks.landmark[right_eye_index].y * h)
            
            # Calculate pixel distance between left and right eye
            pixel_eye_distance = ((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2) ** 0.5
            
            # Calculate focal length
            focal_length = (pixel_eye_distance * known_distance_from_camera) / real_world_eye_distance
            print(f"Focal Length: {focal_length:.2f} pixels")

    # Display the frame with landmarks
    cv2.imshow("Focal Length Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
