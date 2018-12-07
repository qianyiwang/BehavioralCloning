import csv
import cv2
import os
import numpy as np

# data processing
images = []
measurements = []
LOG_PATH = 'simulation-data-1/driving_log.csv'
IMG_PATH = 'simulation-data-1/IMG'
len_images = len(os.listdir(IMG_PATH))
lines = []
with open(LOG_PATH) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        cen_source_path = os.path.join(IMG_PATH, line[0].split('/')[-1]) # index 0 is center images
        left_source_path = os.path.join(IMG_PATH, line[1].split('/')[-1])
        right_source_path = os.path.join(IMG_PATH, line[2].split('/')[-1])
        img_cen = cv2.imread(cen_source_path)
        img_left = cv2.imread(left_source_path)
        img_right = cv2.imread(right_source_path)
        images.extend([img_cen, img_left, img_right])
        measurements.extend([steering_center, steering_left, steering_right])
# add augmented training data
augmented_imgs, augmented_measurements = [], []
for img, measure in zip(images, measurements):
    augmented_imgs.append(img)
    augmented_imgs.append(np.fliplr(img))
    augmented_measurements.append(measure)
    augmented_measurements.append(-measure)

x_train = np.array(augmented_imgs)
y_train = np.array(augmented_measurements)

# build model
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
model = Sequential()
model.add(Lambda(lambda x: x/255,  input_shape=(160, 320, 3)))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(84))
model.add(Dense(1)) # just one output node without and activation function can do regression instead of classification
# compile the model
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')
exit()

