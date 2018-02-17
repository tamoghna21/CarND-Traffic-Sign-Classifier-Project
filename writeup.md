# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/x_train_1000.jpg.jpg "Visualization"
[image2]: ./examples/x_train_1000_grayscale.jpg "Grayscaling"
[image3]: ./examples/x_train_1000_noisy.jpg "Random Noise"
[image4]: ./traffic_signs_web/sign1.jpg "Traffic Sign 1"
[image5]: ./traffic_signs_web/sign2.jpg "Traffic Sign 2"
[image6]: ./traffic_signs_web/sign3.jpg "Traffic Sign 3"
[image7]: ./traffic_signs_web/sign4.jpg "Traffic Sign 4"
[image8]: ./traffic_signs_web/sign5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tamoghna21/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is an image from training dataset(corresponding label is 36:Go straight or right)

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because that gives the Neural network unique features to extract.

Here is the grayscaled version of the same image shown above:

![alt text][image2]

Then I normalized the image data because normalization makes convergence faster and helps avoid local minima of cost function. 

I decided to generate additional data because That helps in generalization.


To add more data to the the data set, I added noise by applying cv2.randn() to each image. Other ways might be to skew the images by applying cv2.warpPerspective() 

Here is a noisy version of the same image shown above:

![alt text][image3]

The difference between the original data set and the augmented data set is that the augmented dataset contains all the noisy versions of the original image data. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x32   |
| RELU					|												
| Max pooling	      	| 2x2 stride,  outputs 6x6x32 				
| Flatten layer	      	| outputs 1152 				
| Fully connected		| 1152 x 400        							|
| RELU					|												
| Dropout Layrer		|												
| Fully connected		| 400 x 200        							|
| RELU					|												
| Dropout Layrer		|												
| Softmax				| fully connected layer: 200 x 43       		|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used learning rate of 0.001 and AdamOptimizer.
I trained the model for 20 EPOCHs because, I saw the validation set accuracy was nearly constant by 20 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 
* validation set accuracy of 0.956
* test set accuracy of 0.937

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
           Architecture more or less remained same. I increased the filter depth of the first and second convolution layers,              because that increased the accuracy, because more filter depth means more features detected. Also drop out was                added to avoid overfitting.
* What were some problems with the initial architecture?
           Initially the validation accuracy remained more or less 0.93. I wanted to make it a little higher.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
            I added dropout to avoid overfitting. Also, random noise was added in the traing samples and appened with the                   original training dataset.
* Which parameters were tuned? How were they adjusted and why?
            Filter depth of first and second convolutional layers increased to detect more features. Also, the filter size of               the 2nd convolutional layer changed from 5x 5 to 3x3, because the input image of the 2nd convolutional layer was               14x14x12 and applying a filter of 5x 5 seemed to be too big for a smaller image.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
             Dropout layer helps in generalization. Without dropout layer the network may fit very good on the traing set but                may fail in generalization. 

If a well known architecture was chosen:
* What architecture was chosen?
    I modified the LeNet architecture in the current project.
* Why did you believe it would be relevant to the traffic sign application?
   Lenet architecture is good for MNIST dataset and I modified the input images into grayscale.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
   Validation accuracy is 0.95 and test set accuracy is 0.94. Both being high and similar means the model is doing good           generalization.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third and fourth images are not classified properly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory  | Roundabout mandatory   						| 
| No entry     			| No entry 										|
| General caution		| Slippery road									|
| Pedestrians	      	| Bumpy road					 				|
| Road work			    | Road work      						     	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is  mostly sure that this is a roundabout sign (probability of 0.9), and the image does contain a roundabout sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9        			| Roundabout   									| 
| .00003     			| End of speed limit (80km/h) 					|
| .00002				| Right-of-way at the next intersection			|
| .00001	      		| Vehicles over 3.5 metric tons prohibited		|
| .000001				| End of all speed and passing limits      		|

For the second image, the model is  sure that this is a No entry sign (probability of 1), and the image does contain a No entry sign. The softmax probabilities of next 4 candidates are too small.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


