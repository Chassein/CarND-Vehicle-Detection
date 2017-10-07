
# Vehicle Detection and Tracking
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In the following, I describe my solution to the Vehicle Detection and Tracking Project of the Udacity - Self-Driving Car NanoDegree. I will consider the [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points individually and describe how I implemented them.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/sliding_windows.png
[image2]: ./output_images/heat_raw.png
[image3]: ./output_images/heat_blure.png
[image4]: ./output_images/heat_with_boxes.png
[video1]: ./project_solution.mp4


## Classification

The project description proposes to use handcrafted features like color histograms and HOG combined with an SVM to find cars in the image. I decided to deviate from the proposed method and used a more advanced method instead. I didn't used any handcrafted features. Instead I used the raw pixel data combined with a neural network to find cars. I will explain in the following the architecture, the training, and the performance of the neural network.


### Neural network - Architecture

The input of the neural network is an image with a resolution of 64x64 pixels. The output is a number between 0 and 1 indicating if the image contains a car or not.

**Layers:**
* Conv Layer with size (3,3) and depth 8 with an RELU activation
* Conv Layer with size (3,3) and depth 8 with an RELU activation
* Max Pooling Layer with size (2,2)
* Conv Layer with size (3,3) and depth 16 with an RELU activation
* Conv Layer with size (3,3) and depth 16 with an RELU activation
* Max Pooling Layer with size (2,2)
* Conv Layer with size (3,3) and depth 32 with an RELU activation
* Conv Layer with size (3,3) and depth 32 with an RELU activation
* Max Pooling Layer with size (2,2)
* A fully connected layer with 50 neurons and RELU activation
* A 50% dropout layer
* A fully connected layer with 20 neurons and RELU activation
* A 50% dropout layer
* A fully connected layer with 10 neurons and RELU activation
* A single output neuron  with SIGMOID activation

### Neural network - Training

The complete data set contains 8792 images of cars and 8968 images without cars. I used 20% of the data as validation set. The images are normalized by substracting the mean from each color channel. The adam update rule is used to update the weights. The network is trained for 20 epochs which took less then 200 seconds. As finial model we selected the model after epoch 16, since it performed best on the validation set. 


### Neural network - Performance

The final network achieves on the validation set a loss of 0.0146 and an accuracy of 99.66% (similar performance is found on the training set: loss of 0.0127 and an accuracy of 99.62%).


## Sliding Window Search

I used the sliding window algorithm, which was build during the lecture (I slightly modified the algorithm to provide also windows at the right border of the image). The windows are slided from left to right and from half of the image to the bottom. I used several parameters for the sliding window search. The best results could be obtained with the following values. The size of the sliding window is set to 128 pixels in the x direction and 64 pixels in the y direction. The windows overlap with 60% in both directions. In the next picture I have drawn all windows over which is searched.

![alt text][image1]

---


## Heat Map

To build a heat map, I initialize a array which has the same size as the image and contains only zeros. The following is done for each window proposed by the sliding window search. 

* Resize the image to 64x64 pixels.
* Provide this picture as input to the previously trained neural network.
* Add the activation value of the network to each pixel of the heat map which is contained in the actual window.

Next, I divide all pixels values by 8 (at most 8 windows can overlap, hence he maximum value of a pixel in the heat map is 8) and multiply it with 255. In this way the heat map contains values in the range from 0 to 255. 

![alt text][image2]

To smooth the heat map, I used `cv2.GaussianBlur` with a filter size of (65,65). To stabilize the heat map during the video stream, I combine the last produced heat map (80%) with the heat map found at this frame (20%).

![alt text][image3]

## Finding Boxes from the Heat Map

To find boxes around heated regions of the heat map. I used the following procedure. 

As long as a pixel with value greater than 70 is contained in the heat map, do the following.
* Specify this pixel as the starting point of the new box (with an initial size of 20 in both directions).
* Each rectangle can be enlarged in four directions (top, bottom, right, left). For each of these directions compute the average pixel activation of the heat map. Choose the direction with the maximum average activation. Repeat the rectangle enlargement until the average activation is less than 55.
* Set the activation value of all pixels inside the recently found rectangle to 0.

Note: All small boxes with size less than 1000 pixels are discarded.

![alt text][image4]




## Finding Cars from Boxes

The next step is to use the created boxes to identify cars. To this we create a `car` class. The `car` class contains a box and a believe value. If the believe value is above 0.5 it is assumed that the found boxes indicate indeed a car and the box is drawn on the screen. 

During the processing of the video a list is maintained which contains all currently existing `car` objects.

I loop over all boxes found from the heat map. Assume to have a fixed box B. I compute the euclidean distance of B to the boxes of all existing `car` objects. If the box of one `car` object C is sufficiently close, I do the following: 

Update the believe of this C by the following formula  (Note: This formula converges to a belive of 1 if it is applied multiple times): <br><br>

<center>Believe of C = 0.95 * Belive of C + 0.05</center>

Further, I update the box of C:

<center>Box of C = 0.9 * Box of C + 0.1 * Box </center> <br><br>

If Box B was not close enough to any existing `car` object. I create a new `car` object with box B and belive of 0.05.  

Further, I check for all existing `car` objects C if a box was found which is close to C. If this is not the case the believe of C is updated: <br><br>

<center> Believe of C = 0.975 * Belive of C </center>


## Result

To produce the final video I combined the described pipeline with the pipeline of the last project which was used to detect lane lines. The complete pipeline runs at about 2 frames/second, the tracking pipeline alone runs at about 4.5 frames/second on a medium desktop PC with an Nvidia GTX 670.

Have a look at my [final result](./project_solution.mp4).

---

### Discussion

The pipeline fails to identifies cars correctly if they overlap on the image. This is due to the representation of a car as a 2 dimensional box object. For an improved pipeline which can also handle such cases it is necessary to create a 3 dimensional environment of the car but this is beyond the scope of this project.

The neural network which is used to identify if a part of the image contains a car shows almost perfect performance on the validation set. However, the performance in the video shows room for improvement. Using either a larger training set, or finetuning a neural network which was pretrained on a larger data set may prove to be beneficial. 

To further improve the classification accuracy one might consider to change the size of the sliding windows depending on the location of the image.

State of the art methods don't rely on the proposed sliding window and box finding algorithms. Instead, a single neural network is used to form region proposals and to classify these regions.
