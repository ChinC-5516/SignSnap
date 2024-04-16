import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 50
collection_time_per_class = 10  # Time in seconds for collecting data for each class
buffer_time = 5  # Buffer time between collecting data for different classes

cap = cv2.VideoCapture(0)  # Use index 0 for the default camera (laptop's built-in camera)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Function to display a countdown message
def countdown_message(class_index, seconds):
    print(f'Collecting data for class {class_index} in', end=' ')
    for i in range(seconds, 0, -1):
        print(i, end=' ')
        time.sleep(1)
    print()

done = False
while True:
    ret, frame = cap.read()
    cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Display countdown message before collecting data for each class
    countdown_message(j, buffer_time)

for j in range(number_of_classes):
    print('Collecting data for class', j)
    start_time = time.time()  # Start time for collecting data for the current class
    
    while time.time() - start_time < collection_time_per_class:
        ret, frame = cap.read()
        cv2.putText(frame, 'Collecting data for class {}'.format(j), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        # Save the images for both hands
        for i, hand in enumerate(['left', 'right']):
            ret, frame = cap.read()
            cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{hand}_{time.time()}.jpg'), frame)

    print('Data collected for class', j)
    
    # Display message for data collection completion
    print('Data collection for class', j, 'completed.')
    
    # Display buffer time
    print('Waiting for', buffer_time, 'seconds before collecting data for the next class...')
    time.sleep(buffer_time)

cap.release()
cv2.destroyAllWindows()
