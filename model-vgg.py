import csv
import cv2
import os
import numpy as np

# data processing
images = []
measurements = []
LOG_PATH = 'simulation-data-2/driving_log.csv'
IMG_PATH = 'simulation-data-2/IMG'
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

# Load the pre-trained VGG model
from keras.applications import VGG16
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(160, 320, 3))

# Freeze the required layers
# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
  
# Create the model
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
# Add the vgg convolutional base model
model.add(vgg_conv)
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
# Show a summary of the model. Check the number of trainable parameters
model.summary()

# TODO: greate train and validation data generator

# Compile the model
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])
# Train the model
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.samples/train_generator.batch_size ,
#     epochs=5,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples/validation_generator.batch_size,
#     verbose=1)
model.compile(loss='mse', optimizer='adam')
history = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1) 
# Save the model
model.save('model_vgg_data2.h5')

# visualization
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(acc))
# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()