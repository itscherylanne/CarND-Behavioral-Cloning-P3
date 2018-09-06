# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./examples/ModelArchitecture.png "Model Visualization"
[image2]: ./examples/00_RawHistogram.png "Raw Data"
[image3]: ./examples/01_FilteredHistogram.png "Filtered"
[image4]: ./examples/02_LeftRightAddedHistogram.png "Left and Right Cameras Added"
[image5]: ./examples/03_FlippedDataHistogram.png "Flipped Added"
[image6]: ./examples/normal_example.jpg "Normal Image"
[image7]: ./examples/flipped_example.jpg "Flipped Image"


Overview
---
This repository contains starting files for my submission of the Behavioral Cloning Project.

This project demonstrates the topics I learned about deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated, and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

The project uses a simulator (provided by Udacity) where one can steer a car around a track for data collection. This simulator was used to collect image data and steering angles to train a neural network. In addition, this simulator then use this model to drive the car autonomously around the track.

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network (CNN) (This is a copy of model-track1.h5)
* model-track1.h5 trained CNN using Track-1 data only
* model-track2.h5 trained CNN using Track-2 data only
* model-track1and2.h5 trained CNN using Track-1 and Track-2 data
* README.md summarizing the results

---
### Model Architecture


My model consists of a convolution neural network similar to the model by NVIDIA the autonomous car group (model.py lines 139-161). I chose this model over LeNet, AlexNet, and fully-connected convolution networks was because the NVIDIA model would converge in less epochs and was faster to train due to striding.


A summary of the model from model.summary() may be found below.
```ssh
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 220, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 108, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 108, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 52, 36)    21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 14, 52, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 24, 48)     43248       dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 5, 24, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 22, 64)     27712       dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 3, 22, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 20, 64)     36928       dropout_4[0][0]                  
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 1, 20, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1280)          0           dropout_5[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           128100      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 265,019
Trainable params: 265,019
Non-trainable params: 0
____________________________________________________________________________________________________


```


#### Attempts to reduce overfitting in the model

Dropouts were used to prevent the model from overfitting. A dropout of 0.1 was used for every convolution layer of the model. This was implemented on lines 147, 149, 151, 153, and 155 of model.py. When adding this to the model, the car would sway less from side to side when there was straight road on the track.

#### Model parameter tuning

The model used an Adam optimizer with an MSE loss function, so the learning rate was not tuned manually (model.py line 25).


### Model Architecture - Final Architecture

The final model architecture can be summarized in the following graphic.

![alt text][image1]

### Creation of the Training Set & Training Process
The model was created by driving on Track-1 for 5 laps. The 5 laps consisted of (1) maintaining center-of-lane driving and (2)recovery driving during turns. There were 10,228 instances collected.

<i><b>Note:</b>  I repeated this process on Track-2 for 3 laps in order to get more data points. However, this data was omitted from the model as it made the model perform worse (it would complete the track but the car would drive side-to-side).</i>


The imagery was collected utilizing a keyboard. So there were a lot of instances when the steering angle was 0 degrees.

![alt text][image2]

To prevent the model from biasing to have a steering angle of 0 degrees, the input data was filtered (Don't worry, we will add images where the camera angle is 0 degrees later.) Additionally, frames when the speed was less than 0.5 mph were filtered out as it does not exhibit real driving behavior. The filter on steering angle and speed may be found on line 24 of model.py. From filtering, the data was reduced to 2,981 data points.

![alt text][image3]


To smooth out the gap in the data due to the lack of 0-degree steering angles, the left and right camera data  was added. After some trial and error, a delta of +/- 0.2 degrees was accounted for in the steering angle. The increased data set to 8,943 data points.

![alt text][image4]

Track-1 was contained more left turns than right turns. In order to remove the bias towards left-turns, the dataset was augmented by adding mirrored images (horizontally flipping) and negating the steering angle associated with the image (multiply by negative 1).


![alt text][image6]
![alt text][image7]

After the collection, filtering, and augmentation process, I had <b>17,886</b> number of data points.

![alt text][image5]

I then preprocessed this data by ensuring the colorspaces are trained in RGB (versus BGR, the default of OpenCV). <i><b>Note:</b> The imagery is normalized and zero-centered within the keras Model. Additionally, the image is cropped to be 200 px wide and 66 px tall within the keras model. </i>

I finally randomly shuffled the data set and put 3,577 of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the plateauing of the training loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Results
---

#### Training Results
```ssh
Epoch 1/3
2400/2384 [==============================] - 7s - loss: 0.1489 - val_loss: 0.1422
Epoch 2/3
2400/2384 [==============================] - 4s - loss: 0.1295 - val_loss: 0.1231
Epoch 3/3
2400/2384 [==============================] - 4s - loss: 0.1355 - val_loss: 0.1314
```
#### Test Results
Video of the model running on the simulator may be found in the following link(s):

https://youtu.be/LNCRkEktD4g


#### Summary
I received a final training loss of 0.1355 and a validation loss of 0.1314. The model drives the car well on the Track-1. It unfortunately crashes immediately on the second track. Interestingly, a model trained with Track-2 data drives adequately on Track-1.

I believe the model can improve to work on Track-2 by improving the input data. The dataset can improve by collecting more recovery data and by utilizing a better user controller (possibly a game controller or joystick instead of the keyboard). Additional improvements would be by extending the dataset by skewing the perspective of Track-1 so that the road would not be flat-and-level simulate hills and dips as seen on Track-2.

Track-2 contains lots of variation on brightness due to shadows from the environment. The model can improve by transforming the colorspace so that brightness variations can be accommodated.
