# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image11]: ./examples/all_class_traffic_types.jpg "all_class_traffic_types"
[image12]: ./examples/classes_distribution.jpg "classes_distribution"
[image2]: ./examples/src_gray.jpg "Grayscaling"
[image13]: ./examples/all_Aug_class_traffic_types.jpg "all_Aug_class_traffic_types"
[image14]: ./examples/classes_distribution_argu.jpg "classes_distribution_argu"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/train_accuracy_50_128_0.0001.jpg "train_accuracy"
[image5]: ./examples/New_Images.jpg "New_Images"
[image6]: ./examples/predict_images.jpg "predict_images"
[image7]: ./examples/Top_proba_new_images.jpg "Top_proba_new_images"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

 * The size of training set, validtion set, and test set is:

    Training Set:   34799 samples

    Valid Set:     4410 samples

    Test Set:      12630 samples

 * The shape of a traffic sign image is 

    Image Shape: (32, 32, 3)

 * The number of unique classes/labels in the data set is 

    43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image11]

class contribution:

![alt text][image12]


## Design and Test a Model Architecture

### step1: Pre-process the Data Set

#### 1.As a first step, I decided to convert the images to grayscale because the gray image works well in classification, and reduce the amount of calculation. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

#### 2.As a last step, I normalized the image data because normalized data is easier to converge in training.



#### 3.In addition, i decided to do data argumentation  because the data is unbalanced and samples set if too small.

To add more data to the the data set, I used the following techniques 

•	Slight random rotations

•	Slight random scale

•	Slight random shift

Here is an example of some augmented image:

![alt text][image13]

after argumentation, sample set's contribution shows below:

![alt text][image14]

 * The size of training set, validtion set, and test set is:

    Training Set:   129000 samples

    Valid Set:     4410 samples

    Test Set:      12630 samples

 * The shape of a traffic sign image is 

    Image Shape: (32, 32, 3)

 * The number of unique classes/labels in the data set is 

    43
    
### step2:  Design model architecture

my final model architecture looks like below, consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3x1       |1x1 stride, valid padding,outputs 30x30x8    |
| RELU              |                               |
| Dropout           | 0.5|
| Convolution 3x3x8     	| 1x1 stride, valid padding, outputs 28x28x26 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x26 				|
| Dropout           | 0.5|
| Convolution 3x3x26	    | 1x1 stride, valid padding, outputs 12x12x60   |
| RELU             |                               |
| Max pooling        | 2x2 stride, outputs  6x6x60
| Fully connected		| inputs 2160, outputs 400        									|
| RELU             |                               |
| Fully connected     | inputs 400, outputs 120                           |
| RELU             |                               |
| Fully connected     | intputs 120, outputs 84                      |
| RELU             |                               |
| Fully connected     | inputs 84, outputs 43                      |
| Softmax				| 43x1        									|

 


### step3: Train model. 

To train the model, I used an AdamOptimizer, and  hyperparameters shows below

EPOCHS = 50

BATCH_SIZE = 128

rate = 0.0009

when epoch is bigger than 10, set learn rate to 0.0001.


My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.946
* test set accuracy of 0.937

validation set accuracy shows below:

![alt text][image4]

#### to train model, i take some steps:

First, i use LeNet-5 to train model on source train set, validation accuracy is about 0.87.

Then, to inprove the accuracy, i modify the LeNet Model, add a convolution layer and a fully connected layer. the accuracy on validation is over 0.93, however, predict new images is too bad.

analysis predict images and train images, I find some class images is to small,so I do argumentation that make all class type images's count is 3000.

In addition, the result of train shows the accuracy of train is 1.0, but test accuracy 's not over 0.93. So,I add  two Dropout layer to model architecture.

then, I did lots of experiments, adjusted the hyperparameters. Finally, i find the model works well when set EPOCHS=50,BATCH_SIZE = 128,rate = 0.0009.





### step4: Test a Model on New Images

#### 1. download ten German traffic signs  on the web;

Here are five German traffic signs that I found on the web:

![alt text][image5]

question:

1. the backgroud of images is pure white, are different frome the images in train set.

2.The image Speed limit(30km/h) might be difficult to classify because the sample of "Speed limit(30km/h)" in X_train is too little.

#### 2. Here are the results of the prediction:


Here are the results of the prediction:

| Image			        |     Prediction	        					|  result|
|:---------------------:|:---------------------------------------------:|:---------------:|
| Yield     			| Yield 									| true  | 
| Speed limit (30km/h)  | Speed limit (30km/h)   					| true |
| Go straight or left   | Go straight or left 						| true  |
| General caution      | General caution 						| true  |
| Wild animals crossing | Wild animals crossing 					| true  |

![alt text][image6]
The model was able to correctly predict 5 of the 5 traffic signs, which gives an accuracy of 100%. 


#### 3.  the softmax probabilities for each prediction. 


The top five soft max probabilities were

![alt text][image7]

my question: why the probability is 1.0?



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Can you give me some examples?


```python

```
