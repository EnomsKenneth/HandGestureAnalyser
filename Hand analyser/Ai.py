import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open Webcam.")
    hands.close()
    raise SystemExit
print("Hand Gesture Analyser is up and running!!")
print("Press Esc to exit")
# Process the webcam video
try:
    while True:
        # Capture one frame
        success, frame = cap.read()

        if not success:
            print("Failed to capture video frame.")
            break

        # Mirror the video like a selfie camera
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(rgb_frame)

        # Draw hand landmarks and connections
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Display the live webcam feed
        cv2.imshow("Hand Gesture Analyser", frame)

        # Exit when Esc is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Hand Gesture Analyser stopped.")