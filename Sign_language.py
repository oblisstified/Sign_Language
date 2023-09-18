
import cv2
import os
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_landmark_data(image):
    data_aux = []
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    
    #each landmark has its own x,y value
    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append([x,y])
    return data_aux



vid = cv2.VideoCapture(0)
img_counter = 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
mpDraw=mp.solutions.drawing_utils

ASSETS_DIR = 'Sign_Language/Assets'
hand_landmarks_data = []
labels = []

for dir_ in os.listdir(ASSETS_DIR):
    for img_path in os.listdir(os.path.join(ASSETS_DIR,dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(ASSETS_DIR,dir_,img_path))
        hand_landmarks_data.append(get_landmark_data(img))
        labels.append(ord(dir_)-66) #change to -65 for other than binary

data = tf.data.Dataset.from_tensor_slices((hand_landmarks_data, labels)).batch(32).shuffle(True)

train_size = int(len(hand_landmarks_data) * 0.7)
val_size = int(len(hand_landmarks_data) * 0.2)
test_size = len(hand_landmarks_data) - train_size - val_size

train_data = data.take(train_size)
val_data = data.skip(train_size).take(val_size)
test_data = data.skip(train_size+val_size).take(test_size)

model = keras.Sequential([
    layers.Input(shape=(21, 2)),  # 21 hand landmarks with x, y
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_data ,epochs=400, validation_data=val_data)



while(True): 
    ret, frame = vid.read()
    x,y,c = frame.shape
      
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)
    # Post-process the result
    if result.multi_hand_landmarks:

        hand_landmarks = get_landmark_data(frame)
      
        mpDraw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        single_sample = np.expand_dims(hand_landmarks, axis=0) 
        yhat = model.predict(np.array(single_sample))
        print(yhat)
        if yhat > 0.999: 
            print(f'Predicted class is C')
        elif yhat<0.001:
            print(f'Predicted class is B')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        
        img_name = f"test_image{img_counter}.jpg"
        cv2.imwrite(img_name,frame)
        img_counter+=1
        print("image saved")
vid.release()
cv2.destroyAllWindows()