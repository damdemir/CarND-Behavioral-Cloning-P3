# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_2020_05_06_17_32_28_057.jpg "Center View"
[image2]: ./images/left_2020_05_06_17_32_28_057.jpg "Left View"
[image3]: ./images/right_2020_05_06_17_32_28_057.jpg "Right View"
[image4]: ./images/OriginalandFlipped.png "Original and Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
If the run is going to be recorded then a folder name should be selected like;
```sh
python drive.py model.h5 run1
```
After, the images of autonomous driving in run1 folder are converted to video like below;
```sh
python video.py run1
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 32 and 128 (clone.py lines 67-77) 

The model includes RELU layers to introduce nonlinearity (code line 70-72), and the data is normalized in the model using a Keras lambda layer (code line 68). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 56-57). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 79).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to achieve the autonomous driving which takes images and learn then ouputs a steering angle.

My first step was to use a convolution neural network model. I thought this model might be appropriate because there are lots of training data and it is for neural network problem.

Then I split the training and validation data set to reduce the overfitting.
I drove the vehicle 2 laps and to do recover the vehicle itself i repeated some hard curves with different initial position getting the center of lane.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 67-77) consisted of a convolution neural network with the following layers and layer sizes:
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Cropping     	| 70x25 	|
| Convolution 5x5	    | activation = relu     									|
| Max pooling				|     									|
| Convolution 5x5	    | activation = relu     									|
| Max pooling				|     									|
|	Flatten					|									|
|	Dense					|	outputs 120											|
|	Dense					|	outputs 84											|
|	Dense					|	outputs 1										|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. It makes the car getting back to center.

![alt text][image1]
![alt text][image2]
![alt text][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generalize the data. For example, here is an image that has then been flipped:

![alt text][image4]


After the collection process, I had 21870 number of data points 7290 images represents for center, 7290 images represents left images,and 7290 images represents the right images. I then preprocessed this data by flipping images and taking reverse of steering measurement. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. That means training data has 5382 lines including center, left, and right.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.


