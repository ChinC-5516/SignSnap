import cv2
import numpy as np
import mediapipe as mp
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load trained SVM model
with open('svm_model.pickle', 'rb') as f:
    svm_model = pickle.load(f)

# Map class indices to class names
class_names = {
    0: "gojo",
    1: "chinmay",
    2: "All Good",
}

# Load images for each class and resize them
class_images = {
    "gojo": cv2.resize(cv2.imread("gojo.jpg"), (200, 200)),
    "chinmay": cv2.resize(cv2.imread("chinmay.jpg"), (200, 200)),
    "All Good": cv2.resize(cv2.imread("all_good.jpg"), (200, 200)),
}

cap = cv2.VideoCapture(0)  # Use index 0 for the default camera

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process hand landmarks
        results = hands.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            # Use only the landmarks of the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [landmark.x for landmark in hand_landmarks.landmark]
            landmarks += [landmark.y for landmark in hand_landmarks.landmark]

            # Predict the class of hand sign
            predicted_class = svm_model.predict([landmarks])[0]

            # Get class name
            class_name = class_names[int(predicted_class)]  # Convert predicted_class to int

            # Print class name in console
            print("Detected hand sign:", class_name)

            # Display class name
            cv2.putText(image, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display corresponding image for the predicted class
            if class_name in class_images:
                # Show the resized image
                cv2.imshow("Class Image", class_images[class_name])

        cv2.imshow('Hand Sign Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
