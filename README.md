**Traffic Sign Recognition**
---
[//]: # (Image References)

[image1]: ./ReadMe_images/Distribution_y_train.png "Distribution of Y train"
[image2]: ./ReadMe_images/Distribution_y_validation.png "Distribution of Y validation"
[image3]: ./ReadMe_images/Distribution_y_test.png "Distribution of Y test"
[image4]: ./ReadMe_images/trafficsigns.png "traffic signs"
[image5]: ./ReadMe_images/5_gray.png "grayscale"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 Grayscale image   							| 
| Convolution 5x5     	 | 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
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


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
