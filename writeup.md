## Writeup Template
### Vehicle Detection Project

---

In this project, my goal was to write a software pipeline to detect vehicles in a video (starting first with the test_video.mp4 and later implement on full project_video.mp4); the pipeline is presented in detail in writeup of the project.

The project has three main steps:
- Histogram of Oriented Gradients (HOG)  
    - we have been provided with a set of images: 8968 various traffic images, 8792 images with cars;images are color, 64x64 pixels  
    - using HOG and other techniques, extract features of these images  
    - separate the images in train/test and train a SVM classifier  
- Sliding Window Search  
    - implemented a sliding window search and classify each window as vehicle or non-vehicle  
    - run the function first on test images and afterwards against project video 
- Video Implementation  
    - output a video with the detected vehicles positions drawn as bounding boxes  
    - implement a robust method to avoid false positives (could be a heat map showing the location of repeat detections)

In the end the results are interpreted and discussed.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[image8]: ./output_images/parameters_search.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup 

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

The present document represents the writeup.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the cells 1 to 16 of the Ipythin notebook in this repository, *Vehicle_Detection.ipynb*

The implementatin starts with importing the relevant modules for this project; all images were place in the same directory on a local drive; non-car imags have been renamed starting with 'traffic'; this helped separate them in car and non-car lists; the dataset has 8792 car images and 8968 non-car images; the data set is well balanced and it doesn't need augmentation.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

During the lecture I put together the code discussed and tested different combinations changing only one parameter a time, on a range of values; the best accuracy would indicate wich value to keep and went ahead to test another parameter; below you can see the efect of each parameter, and in bold the one kept for final run.
![alt text][image8]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In cell 9, the features of each of the car and non-car list are extracted; these are used to train the linear SVM model; first the are normalized using the StandardScaler, cel 12; then in cell 14 the model is initiated and trained; the test accuracy was 98.51%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The next section is the main part of the project: using small windows, each image is searched; data from each window is than tested againts the trained model and a decision is made if it conains a car or not; using several windows we can detect a car in more than one window; this is particulary helpful to reduce false positive cases; using a heat filter we will extract the locations where we the a car was detected more than 5 times, ignoring all other windows.

I searched with 3 window sizes: 90px, 96px and 112px; these were selected after a few trial and error tests:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult part of the project was elimating the false positives; event we had good accuracy on the test, in the project this didn't seem good enough; this is a sign the training model had overfit the training data;
One way to improve the project would be to try another traing model, maybe a neural network.

