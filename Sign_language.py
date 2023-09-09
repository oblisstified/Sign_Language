
import cv2
import os
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


  
vid = cv2.VideoCapture(0)
img_counter = 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)

ASSETS_DIR = 'Sign_Language/Assets'
hand_landmarks_data = []
labels = []

for dir_ in os.listdir(ASSETS_DIR):
    for img_path in os.listdir(os.path.join(ASSETS_DIR,dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(ASSETS_DIR,dir_,img_path))
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        #each landmark has its own x,y value
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)


            hand_landmarks_data.append(data_aux)
            labels.append(dir_)
print(hand_landmarks_data[0],len(hand_landmarks_data[120]),len(hand_landmarks_data[100]),len(hand_landmarks_data[20]))

#builds image dataset that builds labels, classes, resize images,batch images tyo size 32
# data = tf.keras.utils.image_dataset_from_directory('Sign_Language/Assets')

data = tf.data.Dataset.from_tensor_slices((hand_landmarks_data,labels)).batch(32).shuffle(True)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
data = data.map(lambda x,y: (x/255, y)) #do i need to do this if all the values are already below 1
data.as_numpy_iterator().next()

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
print(data.element_spec[0].shape)

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(None,42)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()
    
# while(True):
      
    
#     ret, frame = vid.read()
  
 
#     cv2.imshow('frame', frame)
      
   
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         print("hello")
#         break

#     if cv2.waitKey(0) & 0xFF == ord('s'):
        
#         img_name = f"test_image{img_counter}.jpg"
#         cv2.imwrite(img_name,frame)
#         img_counter+=1
#         print("image saved")
# vid.release()
# cv2.destroyAllWindows()