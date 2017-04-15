**Traffic Sign Recognition**
---
[//]: # (Image References)

[image1]: ./ReadMe_images/Distribution_y_train.png "Distribution of Y train"
[image2]: ./ReadMe_images/Distribution_y_validation.png "Distribution of Y validation"
[image3]: ./ReadMe_images/Distribution_y_test.png "Distribution of Y test"
[image4]: ./ReadMe_images/trafficsigns.png "traffic signs"
[image5]: ./ReadMe_images/5_gray.png "grayscale"
[image6]: ./web_images/1.png "1"
[image7]: ./web_images/2.png "2"
[image8]: ./web_images/3.png "3"
[image9]: ./web_images/4.png "4"
[image10]: ./web_images/5.png "5"

**Build a Traffic Sign Recognition Convolutional Neural Network**

![traffic signs][image4]

#### Tools used
* Python 3.6
* Numpy
* Matplotlib
* Tensorflow
* Jupyter notebook
* AWS GPU Instance

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
Here is a link to my [project code](https://github.com/Mn0491/Traffic_Sign_Classifier_Convolutional_Neural_Network/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Load the Data

I used python and numpy to calculate some the summary statistics of the traffic sign data set

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the training data

![Distribution of Y train][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing
##### Step 1. Grayscale

As a first step, I decided to convert the images to grayscale because color is not a decisive factor for recognizing the traffic signs. The decisive factor for the traffic sign would be the shape as well as the image pattern within that traffic sign. By grayscaling the image it would also increase the CNN's training speed.   

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image5]

##### Step 2. Normalization

As a second step, I normalized the image data to increase the performance of the CNN. Normalizing the data helps treat all the weights "fairly" when doing backpropagation. This helps prevent over compensating a correciton to one weight while under compensating in another

##### Step 3. Shuffling

As a last step, I shuffled the image data to avoid highly correlated batches of traning data. This helps decrease the bias of the network.  

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		      |     Description	        					                 | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 Grayscale image   							             | 
| Convolution 5x5     	 | 1x1 stride, Valid padding, outputs 28x28x6 	  |
| RELU					             |											                                   	|
| Max pooling	2x2      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	      | 1x1 stride, Valid padding, outputs 10x10x16 	|
| RELU					             |												            |
| Max pooling	2x2      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten               | outputs 400        									      |
|	Fully Connected Layer	|	outputs 120											|
| RELU					             |												            |
|	Dropout              	|	70% Keep probability											|
|	Fully Connected Layer	|	outputs 84											|
| RELU					             |												            |
|	Dropout              	|	70% Keep probability											|
|	Fully Connected Layer	|	outputs 43											|


#### 3. Hyperparamters used to train model

To train the model, I used the following hyperparameters:
* preprocessed images
* Adam optimizer 
* Learning reate 0.001
* Batch Size 128
* Epochs 10
* Dropout Keep Probability 0.7

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 93.3%
* test set accuracy of 91.9%

I chose to do an iterative approach on training my neural network. First I chose to use the LeNet architecture. I chose this as I am most familiar with this architecture, and I figured tweaking it would be a good place to start. Using LeNet that gave me an accuracy of about 86% which was not good enough. I decided to experiment with adding 2 drop out layers to prevent the model from overfitting. I decided to go with a 70% keep probability after some researcg online. This worked well as it boosted my validation accuracy rate up to 93%. 
 
### Test a Model on New Images

####1. Images found on web

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The signs with a white background may be hard to predict since most signs has some sort of actual background image of the enciroment.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn Right Ahead      		     | Turn Right Ahead   								 	| 
| 80km/h     			        | 30km/h 										     |
| 20km/h					            | 30km/h            |
| Stop	      		     | Stop					 		  		|
| Roundabout			      | Priority Road      			|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. The test prediction was at 91.5% which performed a lot better. This is not the greatest of all accuracy and it could be do to not augmenting the images for the training pipeline. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			       | Turn Right Ahead   									| 
| .005    				          | Right of way 										|
| .001					| End of speed limit											|
| .001	      			| Beware of Ice					 				|
| .001				    | Stop      							|


For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999         			       | 30km/h   									| 
| .00    				            | 50km/h 										|
| .00				             | End of Speed											|
| .00      		        	| 80km/h					 				|
| .00			            | 60km/h      							|

For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95        			       | 30km/h   									| 
| .04    				            | Turn right ahead 										|
| .0002				             | Yield											|
| .00      		        	| Stop					 				|
| .00			            | 70km/h      							|

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			       | Stop   									| 
| .001    				            | Turn left ahead 										|
| .00			             | Keep Right											|
| .00      		        	| 20km/h					 				|
| .00			            | Turn Right Ahead|

For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .59        			       | Priority Road   									| 
| .28   				            | Keep Right 										|
| .05			             | End of all speed and passing limits											|
| .04      		        	| End of no passing					 				|
| .01			            | Roundabout|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
