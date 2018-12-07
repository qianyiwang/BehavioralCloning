# Behavioral Cloning Project

## The goals/steps of this project are the following:
- Use the simulator to collect data of good driving behavior
- Build, a CNN in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road

## Files included in submission:
- model.py: containing the script to load data, process data, CNN model building, training and validation
- model-vgg.py: containing the script to load data, process data, load a pre-trained VGG model and fine-turned preparemeters
- drive.py for driving thecar in autonomous mode
- video.py for making video based on simulation result
- model.h5 is a trained CNN model
- model_vgg.h5 is a fine-turned model using pre-trained VGG
- model_vgg_data2.h5 is a fine-turned model using pre-trained VGG and another data set

## Data processing
### Read images (center, left, right) path and steering angle from csv file
```python
with open(LOG_PATH) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(row[3])
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        cen_source_path = os.path.join(IMG_PATH, line[0].split('/')[-1]) # index 0 is center images
        left_source_path = os.path.join(IMG_PATH, line[1].split('/')[-1])
        right_source_path = os.path.join(IMG_PATH, line[2].split('/')[-1])
        img_cen = cv2.imread(cen_source_path)
        img_left = cv2.imread(left_source_path)
        img_right = cv2.imread(right_source_path)
        images.extend(img_cen, img_left, img_right)
        measurements.extend(steering_center, steering_left, steering_right)
```

### Add augmented training data -- flip the image like mirror
```python
for img, measure in zip(images, measurements):
  augment_img = np.fliplr(img)
  augment_measure = -measure
```

## Build CNN model
### Design my own model
- Normalize training data by adding Lambda layer
- Add Conv2D lay 1 with 'relu' activation function
- Add MaxPooling2D with default size (2,2)
- Add Conv2D lay 2 with 'relu' activation function
- Add MaxPooling2D with default size (2,2)
- Add Flatten layer
- Add Dense layer with output 128
- Add Dense layer with output 84
- Add Dense layer with output 1 because we need regression but not classification
- Randomly select 20% of training data to do validation and set 5 epoches to train

### Use pre-trained VGG16 model, freeze some layers, add some new layers and fine-turned the model
- Load pre-trained VGG, with default weights from imagenet
```python
from keras.applications import VGG16
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(160, 320, 3))
```
- Freeze the layers except the last 4 layers
```python
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
```
- Add Flatten layer
- Add Dense layer with 1024 output and relu activation function
- Add Dropout layer with 50 dropout probability
- Add Dense layer with output 1 because we need regression but not classification
- Randomly select 20% of training data to do validation and set 3 epoches to train
## Simulation performance analysis
From the video, we can see most of time, the vehicle drives along the right side lane but not in the center of two lanes. I think the reason is I only run the training in one direction and I always drive along right lane. Another reason could be, my simulator from workspace is very stuck. It is hard to collect good data.