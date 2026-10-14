
import cv2
import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    # Calculate distance between two landmarks
    def distance(a, b):
        return math.hypot(
            lm[a].x - lm[b].x,
            lm[a].y - lm[b].y
        )

    # Palm width for scale-independent thresholds
    palm_width = distance(5, 17)

    if palm_width < 0.001:
        return "Gesture not recognized"

    # Check whether fingers are extended
    index_up = distance(8, 5) > distance(6, 5) * 1.15
    middle_up = distance(12, 9) > distance(10, 9) * 1.15
    ring_up = distance(16, 13) > distance(14, 13) * 1.15
    pinky_up = distance(20, 17) > distance(18, 17) * 1.15

    # Check thumb extension
    thumb_extended = distance(4, 5) > palm_width * 0.65

    # Check thumb direction with a margin
    thumb_up = lm[4].y < lm[3].y - palm_width * 0.15
    thumb_down = lm[4].y > lm[3].y + palm_width * 0.15

    # THUMBS UP
    if (
        thumb_extended and thumb_up
        and not index_up and not middle_up
        and not ring_up and not pinky_up
    ):
        return "THUMBS UP!"

    # THUMBS DOWN
    if (
        thumb_extended and thumb_down
        and not index_up and not middle_up
        and not ring_up and not pinky_up
    ):
        return "THUMBS DOWN!"

    # PEACE SIGN
    fingertips_separated = distance(8, 12) > palm_width * 0.25

    if (
        index_up and middle_up
        and not ring_up and not pinky_up
        and fingertips_separated
    ):
        return "PEACE SIGN!"

    return "Gesture not recognized"


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        cap.release()
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create a resizable webcam window
    window_name = "Hand Gesture Analyzer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands:

            print("Hand Gesture Analyzer is running!")
            print("Show your hand to the camera.")
            print("Press Q to exit.")

            while True:
                success, img = cap.read()
                attempt = 0

                # Retry failed camera reads
                while not success and attempt < 5:
                    time.sleep(0.2)
                    success, img = cap.read()
                    attempt += 1

                if not success:
                    print("Failed to read frame.")
                    break

                # Mirror the webcam feed
                img = cv2.flip(img, 1)
                h, w, _ = img.shape

                # Convert BGR to RGB for MediaPipe
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Default message
                display_text = "No hands detected"
                text_color = (0, 0, 255)

                if results.multi_hand_landmarks:
                    detected_gestures = []

                    for hand_landmarks in results.multi_hand_landmarks:

                        # Draw hand landmarks and connections
                        mp_draw.draw_landmarks(
                            img,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                        # Label fingertips
                        finger_tips = {
                            "Thumb": 4,
                            "Index": 8,
                            "Middle": 12,
                            "Ring": 16,
                            "Pinky": 20
                        }

                        for name, index in finger_tips.items():
                            landmark = hand_landmarks.landmark[index]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)

                            cv2.putText(
                                img,
                                name,
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )

                        # Recognize the gesture
                        gesture = detect_gesture(hand_landmarks)
                        detected_gestures.append(gesture)

                    # Display detected gestures
                    display_text = " | ".join(detected_gestures)
                    text_color = (0, 255, 0)

                # Draw a dark background behind the message
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2

                (text_w, text_h), baseline = cv2.getTextSize(
                    display_text, font, font_scale, thickness
                ) 

                text_x = max(10, (w - text_w) // 2)
                text_y = h - 30

                cv2.rectangle(
                    img,
                    (text_x - 12, text_y - text_h - 12),
                    (text_x + text_w + 12, text_y + baseline + 8),
                    (30, 30, 30),
                    cv2.FILLED
                )

                # Show the gesture at the bottom center
                cv2.putText(
                    img,
                    display_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                    cv2.LINE_AA
                )

                # Display webcam feed
                cv2.imshow(window_name, img)

                # Press Q to exit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Hand Gesture Analyzer stopped.")


if __name__ == "__main__":
    main()
