#!/usr/bin/env python
# coding: utf-8

# <img src="https://camo.githubusercontent.com/cc5541fa8bc6f822cb8d5b4261a23f20ec8321f4/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f373036353430312f33393131393438362d34373138653338362d343665632d313165382d396663332d3532353061343965663537302e706e67">
# 
# Dieses Hier mal in DER variante machen: https://www.youtube.com/watch?v=j-3vuBynnOE
#
# ---
# 
# CAR OCCUPANCY PREPROCESSOR MODULE
#
# CORE: Keras
#
# Prerequisite is carOccupancy_Car_PreSelector_V2 - This Module is extracting the Training-Material. 
# In this case here the single Cars. Pictures from front with 1 person or 2 Persons
# 

"""
#next time we will normalize better our footage
    camera_height = 500
    # flip the frame
    frame = cv2.flip(frame, 1)

    # rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height) # landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))
    # add rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (0, 255, 0), 2)
    # show the frame
    cv2.imshow("Capturing frames", frame)
    # # Now crop ROI from the frames, resize it to `399*399` and persist them to

"""

# Import necessary libraries:




import os
import random
from glob import glob

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from keras import preprocessing

get_ipython().run_line_magic('matplotlib', 'inline')


width = 96
height = 96

class_names = ['ONE', 'TWO']


def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.jpg')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)

        images.append(x)
    return images


# For cars we have 2 folders ( see carOccupancy_Car_PreSelector_Vx )

images_type_1 = load_images('./detectedImages/ONE')
images_type_2 = load_images('./detectedImages/TWO')



# Bilder mit ONE /  Bilder anzeigen, die Zuvor vom PRE-Selector erzeugt worden sind
#  So wissen wir, das für den Cassifier alles Bereit ist!

plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(random.choice(images_type_1))
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

# show the plot
plt.show()


# TWO person in car /  Bilder anzeigen, die Zuvor vom PRE-Selector erzeugt worden sind
#  So wissen wir, das für den Cassifier alles Bereit ist!


plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(random.choice(images_type_2))
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[1]))

# show the plot
plt.show()


# ### Prepare images as tensors   <-----------------  T E N S O R 


X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)

print(X_type_1.shape)
print(X_type_2.shape)

# We'll now build one big array containing ALL the images: 


X = np.concatenate((X_type_1, X_type_2), axis=0)


# Remember our color pixels, the ones that went from 0-255? We'll rescale them to be between 0 and 1. This will make the model work better.


X = X / 255.

X.shape


# We need to create a `y_train`, so we'll use `0` to indicate `TYPE_1`, and `1` to indicate `TYPE_2`.


from keras.utils import to_categorical

y_type_1 = [0 for item in enumerate(X_type_1)]
y_type_2 = [1 for item in enumerate(X_type_2)]

y = np.concatenate((y_type_1, y_type_2), axis=0)

y = to_categorical(y, num_classes=len(class_names))

print(y.shape)


# ## Convolutional network configuration
# Let's create a deep network which will learn our occupany/cars and then will try to predict them.


from keras.models import Sequential
#from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

# default parameters
conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2
lr = 0.001

epochs = 30
batch_size = 32
color_channels = 3

def build_model(conv_1_drop=conv_1_drop, conv_2_drop=conv_2_drop,
                dense_1_n=dense_1_n, dense_1_drop=dense_1_drop,
                dense_2_n=dense_2_n, dense_2_drop=dense_2_drop,
                lr=lr):
    model = Sequential()

    model.add(Convolution2D(conv_1, (3, 3),
                            input_shape=(width, height, color_channels),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_1_drop))

    model.add(Convolution2D(conv_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_2_drop))
        
    model.add(Flatten())
        
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))

    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))

    model.add(Dense(len(class_names), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    return model


# In[57]:


import numpy as np
np.random.seed(1) # for reproducibility

# model with base parameters
model = build_model()

model.summary()


# In[58]:


epochs = 20


# In[59]:


model.fit(X, y, epochs=epochs)


# We have just created and trained our first deep network model 🎉
# 
# Let's try it!
# 
# ### Predict using our model
# 
# First we are going to test individual images:

# In[60]:


type_1 = preprocessing.image.load_img('./detectedImages/ONE/car11.jpg',    target_size=(width, height))
plt.imshow(type_1)
plt.show()

type_1_X = np.expand_dims(type_1, axis=0)

predictions = model.predict(type_1_X)

print('The type predicted is: {}'.format(class_names[np.argmax(predictions)]))


# In[61]:


type_2 = preprocessing.image.load_img('./detectedImages/TWO/car1485.jpg',  target_size=(width, height))
plt.imshow(type_2)
plt.show()

type_2_X = np.expand_dims(type_2, axis=0)

predictions = model.predict(type_2_X)

print('The type predicted is: {}'.format(class_names[np.argmax(predictions)]))


## Save your model for future use

model.summary()
# 
# You can now export and save your trained model. From now on, you won't need to re-train it. You can just load it and use it for predictions.
model.save('D:\\PROJECT\\a_UC_vehicleOccupancy\\py\\H5Model\\car_occupancy_detection_front.h5')
get_ipython().run_line_magic('pinfo', 'model.save')









