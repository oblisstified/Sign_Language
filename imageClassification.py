# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import PIL
# import tensorflow as tf
  
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

# import matplotlib.pyplot as plt
  

# import pathlib
  
# dataset_url = "https://storage.googleapis.com/download.\
# tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file(
#     'flower_photos', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)  #path to the dataset

# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count) #prints amount of images in the dataset

# #
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(180, 180),
#     batch_size=32)


# val_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(180,180),
#     batch_size=32)

# class_names = train_ds.class_names
# print(class_names)

# plt.figure(figsize=(10, 10))
  
# for images, labels in train_ds.take(1):
#     for i in range(25):
#         ax = plt.subplot(5, 5, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")



# num_classes = len(class_names)
  
# model = Sequential([
#     layers.Rescaling(1./255, input_shape=(180,180, 3)),
#     layers.Conv2D(16, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(
#                   from_logits=True),
#               metrics=['accuracy'])
# model.summary()


# epochs=10
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

import tensorflow as tf
import os #to navigate through file structures

# os.path.join('data','happy')#this goes into the data directory into the file happy, returns file strcuture data//happy and doesnt matter what os you are using 
# os.listdir('data') #lists everything in the folder data

#limit tensorflow using all of the VRAM on the GPU. loading data will expand and use all the potential VRAM on your machine.This prevents this
# gpus = tf.config.experimental.list_physical_devices('GPU') #shows all the gpus available 
# print(gpus)
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True) #tells tf to limit the amount of memory used so you dont get out of memory errors
import cv2
import imghdr

data_dir = 'data'
image_exts = ['jpeg','jpg','bmp', 'png']

for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
import numpy as np
from matplotlib import pyplot as plt
#builds image dataset that builds labels, classes, resize images,batch images tyo size 32
data = tf.keras.utils.image_dataset_from_directory('data')

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next() #this is the batch of 32 images in an array. batch[0] is the images, batch[1] is the labels. next function gives next batch
#so batch[0].shape gives (32,256,256,3) 32 batches of size 256 by 256 pixrls with 3 layers which is rgb
#batch[1] gives an array of 1s and 0s where 1 is ugly and 0 is pretty
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
#does the transformation whe ndata is fetched. when a batch is loaded, you divide every pixel by 255 and y is unchanged
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

#Split data into training and testing partiton to see that the data isnt overfitted
#so training set is going to be 70% of the data
#training data is used to train the model
#validation data is used to evaluate our model while training?
#test partition is used after the training/at the end
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

#take defines how much data is taken in that particular partition
#skip is gonna skip the batches that have been used by the training/training+validaiton
#this is done so that each step uses different batches
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
#adding convolutional layer  and a maxpooling layer.first layer is going to be an input which is going to be 256 by 256 pixels with 3 channels
#it's going to have 16 filters of size 3 by 3 with a stride of 1(filter moves by 1 pixel each time)
#uses relu activation function
#maxpooling chooses max from 2 by 2 layer and reduces array size by 4
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
#flatetens data down.makes 1D array of the data
model.add(Flatten())
#dense layer os where each neuron is connected to every node in the next layer. they are connected by weights 
#first dense layer has 256 outputs. next one has 1 output . output whi
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#you need to compile this. uses the adam optimizer 
#metric you want to track is accuracy 
#summary to see 
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()


logdir='logs'
#going to log your model training 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#.fit is the training component. uses the trianing data,going to  train for 20 epochs(how many times to go over the training)

history = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result(), re.result(), acc.result())

img=cv2.imread("me.jpg")
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
#expand_dims because the model expects a batch so it puts it in another list
yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)
if yhat > 0.5: 
    print(f'Predicted class is ugly')
else:
    print(f'Predicted class is pretty')

from tensorflow.keras.models import load_model
#saves the model in the models folder and you can load the model using the laod _model function

model.save(os.path.join('models','imageclassifier.h5'))
new_model = load_model('imageclassifier.h5')
new_model.predict(np.expand_dims(resize/255, 0))