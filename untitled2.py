# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:18:56 2020

@author: Shriyashish Mishra
"""

#Importing the Keras Libraries and packages
from keras.models import Sequential #initializes the NN in a sequential manner
from keras.layers import Convolution2D #for the convolutional layers and 2D images--deals with images
from keras.layers import MaxPooling2D #for the Pooling layers
from keras.layers import Flatten #for the flattening layers--converts the pooling features into large connected vectors that act as inputs to the NN
from keras.layers import Dense #adds the fully connected layers into an ANN

#Initialising the CNN
classifier = Sequential() #an object of the class

#Step-1 :- Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape= (64, 64, 3), activation='relu')) 

#Step-2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step-3 Flattening
classifier.add(Flatten())

#Step-4 Full Connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#Step-5 Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting images to CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C://Users//Shriyashish Mishra//Desktop//ml//dataset//training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set= test_datagen.flow_from_directory(
        'C://Users//Shriyashish Mishra//Desktop//ml//dataset//test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        'C://Users//Shriyashish Mishra//Desktop//ml//dataset//training_set',
        samples_per_epoch=8000,
        nb_epoch=25,
        validation_data='C://Users//Shriyashish Mishra//Desktop//ml//dataset//test_set',
        nb_val_samples=2000)

