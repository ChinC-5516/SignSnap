import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Use only the landmarks of the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Process landmarks for the first hand
            landmarks_left = [landmark.x for landmark in hand_landmarks.landmark]

            # Flatten the hand landmarks into 1D lists of coordinates
            landmarks_flat_left = landmarks_left + [landmark.y for landmark in hand_landmarks.landmark]
            
            data.append(landmarks_flat_left)
            labels.append(dir_)

# Convert data and labels to NumPy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Save data and labels to pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
