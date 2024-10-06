import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to process frame and detect hand and body landmarks
def process_frame(frame):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hand landmarks
    hand_results = hands.process(rgb_frame)
    
    # Process the frame and find body pose landmarks
    pose_results = pose.process(rgb_frame)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract hand landmark coordinates
            hand_landmarks_list = []
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_landmarks_list.append((cx, cy))
            
            # Here you would add your Makoton sign classification logic
            # classify_makoton_sign(hand_landmarks_list)
    
    if pose_results.pose_landmarks:
        # Draw the pose landmarks
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract body pose landmark coordinates
        pose_landmarks_list = []
        for lm in pose_results.pose_landmarks.landmark:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            pose_landmarks_list.append((cx, cy))
        
        # Here you can add logic to use the body pose landmarks
        # process_body_pose(pose_landmarks_list)
    
    return frame

# Main loop for video capture
cap = cv2.VideoCapture(1)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Process the frame
    processed_frame = process_frame(frame)
    
    # Display the frame
    cv2.imshow('MediaPipe Hands and Pose', processed_frame)
    
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()