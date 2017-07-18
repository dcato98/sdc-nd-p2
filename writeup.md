
# **Udacity Self Driving Car Nanodegree** 
---

## Project #2: Recognize Traffic Signs
---


The goals of this project are to:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report (this file)

[//]: # (Image References)
[image1]: ./histogram.jpg "Histogram of Classes in Train, Validation, and Test Data Sets"
[image2]: ./test_images/test_image_1.jpg "New Traffic Sign 1"
[image3]: ./test_images/test_image_2.jpg "New Traffic Sign 2"
[image4]: ./test_images/test_image_3.jpg "New Traffic Sign 3"
[image5]: ./test_images/test_image_4.jpg "New Traffic Sign 4"
[image6]: ./test_images/test_image_5.jpg "New Traffic Sign 5"

---
### Writeup

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it! Here is a link to my [project code](https://github.com/dcato98/sdc-nd-p2/blob/master/P2-Submission1.html).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The training set contains 34799 images.
* The the validation set contains 4410 images.
* The test set contains 12630 images.
* Each traffic sign image is 32x32 pixels with a depth of 3, corresponding to the RGB color channels.
* There are 43 unique labels in the data set.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the number of labels in the train, validation, and test data sets. 

![alt text][Histogram of Classes]

### Design and Test a Model Architecture

#### 1. Preprocessing

The only preprocessing step applied to the dataset is normalizing the image. This was accomplished by subtracting 128, then dividing by 128, which bounds the pixel values to floats between -1 and 1. 

I did not convert to grayscale because, intuitively, color is useful in identifying some traffic signs.

I chose not to generate additional data because validation accuracy using only the provided dataset was sufficient, however, this would likely be useful for further improving the model.


#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer         				|     Description	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Input         				| 32x32x3 RGB image   							| 
| Convolution 3x3     			| 1x1 stride, valid padding, 28x28x6 output 	|
| RELU							|												|
| Max pooling	      			| 2x2 stride, 14x14x6 output 					|
| Convolution 3x3     			| 2x2 stride, valid padding, 10x10x16 output 	|
| RELU							|												|
| Max pooling	      			| 2x2 stride, 5x5x16 output 					|
| Fully connected, 40% dropout	| 120 output  									|
| Fully connected				| 43 output   									|
| Softmax						|           									|

This is architecture is based on the LeNet architecture implemented in class. The only difference is the application of dropout to the first fully connected layer during training.


#### 3. Final Model Hyperparameters

Model hyperparameters:
* Optimizer: Adam
* Initial learning rate: 0.001
* Batch size: 128
* Epochs: 20 
* Dropout keep %: 40 (during training only)
* Output Encoding: 1-hot

#### 4. Results and Discussion 

I started with the famous LeNet architecture which was, at one time, the state-of-the-art image recognition model. I calculated a baseline accuracy for this model around 0.90. Since fully connected layers have a tendency to memorize the data set (i.e. overfit) without sufficient regularization, this would seem like a good place to start. As dropout is trivial to add in TensorFlow and doesn't increase training time, I started by adding this to the first fully connected layer. This was enough to push the validation set accuracy above 0.93.

In attempts to further improve accuracy, I tried the following:
* Optimizing the dropout rate
* Adding more epochs
* Adding residual layers
* Adding local contrast normalization

I was surprised to find such a low variation in accuracy between 0.3 and 0.8. Depending on the random seed, any of these were among the top results. I settled on 0.4 because, in my limited trial runs, it was most consistent in producing the best overall validation accuracy, however it is certainly possible that this was due to chance. 

The addition of more epochs increased the validation set accuracy from 0.93 to 0.95 at the cost of taking twice as long to train each model. This is unsurprising, since after 10 epochs, the models often continued to show small improvements in accuracy.

Next, after researching state-of-the-art image recognition techniques, I decided to try two alternative model architectures - residual layers and local contrast normalization. Both of these approaches showed small decreases in accuracy. Perhaps these would be more effective in deeper networks, as the papers show these techniques to work for some deep networks.

**My final model results were:**
* training set accuracy of 0.995
* validation set accuracy of 0.957 
* test set accuracy of 0.947

**Additional questions:**
* What was the first architecture that was tried and why was it chosen?<br>
LeNet architecture - because it was at one time the state-of-the-art model for image recognition and it is small enough to train in a few minutes on my laptop.
* What were some problems with the initial architecture?<br>
Overfitting in the fully connected layers, then underfitting (after adding dropout).
* How was the architecture adjusted and why was it adjusted?<br>
Added dropout to the first fully connected layer to try to address overfitting, then increased the number of epochs to try to address underfitting.
* Which parameters were tuned? How were they adjusted and why?<br>
Number of epochs - doubled this parameter from 10 to 20 because, after adding dropout, the loss was still dropping and the validation accuracy continued to improve.
* What are some of the important design choices and why were they chosen?<br>
With 3072 inputs (32x32x3), a significant amount of regularization is important to prevent models from overfitting. Both convolution layers and dropout have a regularization effect by imposing additional limitations during the training of the model. Convolution layers also take advantage of the inherent spatial property of images (i.e. two pixels very close one another are more likely to result in a useful feature than two pixels far apart).
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?<br>
While the test accuracy is slightly lower than the validation accuracy, indicating some overfitting, the test accuracy is still sufficiently high to produce good results.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The first and fourth images might be difficult to classify because they were not in the original data set. Otherwise, they should be relatively easy to classify.

The second image might be difficult to classify because the sign is at a sharp upwards-facing angle.

The third image might be difficult to classify because, instead of being a standard 'Go straight or right' sign, it is part of a larger interstate direction sign and the straight arrow is longer than the standard sign.

The fifth image might be difficult to classify because the sign takes up a smaller portion of the image compared to most images in the dataset and it is not centered.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        				|     Prediction	        			| 
|:-------------------------------------:|:-------------------------------------:| 
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Speed limit (80km/h)					| Speed limit (50km/h)					|
| Go straight or right					| Go straight or right					|
| General caution						| General caution						| 
| Yield									| Road work								| 

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares worse than the accuracy on the test set of 0.943.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very confident that this is a right-of-way sign (probability of 1.0), and it is correct. The top five soft max probabilities are

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 1.0         			| Right-of-way at the next intersection	| 
| 0.0     				| Beware of ice/snow					|
| 0.0					| Pedestrians							|
| 0.0	      			| Children crossing				 		|
| 0.0				    | Roundabout mandatory 					|

For the second image, the model is very confident that this is a 80km/h speed limit sign (probability of 0.99), and it is close, but not correct. The correct answer is the 3rd prediction. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (80km/h)							| 
| .01     				| Speed limit (60km/h)							|
| .00					| Speed limit (50km/h)							|
| .00	      			| No passing for vehicles over 3.5 metric tons	|
| .00				    | Stop 											|

For the third image, the model is relatively confident that this is a go straight or right sign (probability of 0.96), and it is correct. The top five soft max probabilities are

| Probability         	|     Prediction	   	| 
|:---------------------:|:---------------------:| 
| .96         			| Go straight or right	| 
| .04     				| Keep right			|
| .00					| Roundabout mandatory	|
| .00	      			| Turn left ahead		|
| .00				    | End of no passing 	|

For the fourth image, the model is very confident that this is a general caution sign (probability of 1.0), and it is correct. The top five soft max probabilities are

| Probability         	|     Prediction	   		| 
|:---------------------:|:-------------------------:| 
| 1.0         			| General caution			| 
| 0.0     				| Pedestrians				|
| 0.0					| Traffic signals			|
| 0.0	      			| Road narrows on the right	|
| 0.0				    | Go straight or left 		|

For the fifth image, the model is very confident that this is a road work sign (probability of 1.0), and it is incorrect. The correct answer is the 2nd prediction. The top five soft max probabilities are

| Probability         	|     Prediction				| 
|:---------------------:|:-----------------------------:| 
| 1.0         			| Road work						| 
| 0.0     				| Yield							|
| 0.0					| Slippery road					|
| 0.0	      			| Dangerous curve to the right	|
| 0.0				    | No passing					|
