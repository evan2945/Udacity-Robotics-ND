## Project: Object Recognition

In this project, code was written to identify objects sitting on a table utilizing data from a RGB-D camera
-----------------------------------------------------------------------------------------------------------
[//]: # (Image References)

[image1]: ./pick_list_1_svm_results.png
[image2]: ./pick_list_1_obj_recog.png
[image3]: ./pick_list_2_svm_results.png
[image4]: ./pick_list_2_obj_recog.png
[image5]: ./pick_list_3_svm_results.png
[image6]: ./pick_list_3_obj_recog.png

## Objective
In this project, we are solving a perception problem in order to identify and locate target objects in a busy environment.
A 3D map of the environment was created using data from a RGB-D camera. This map allows the robot to visualize the area and
plan accordingly. Many steps were taken to complete this process, all of which are outlined and discussed below.

### Processes completed to accomplish object recognition

#### Converting ROS messages to PCL data
The first step in the process was converting the data received from the RGB-D camera (through ROS messages) to Point Cloud data.
Python has a library named PCL (Point-Cloud Library) that handles this, allowing us to easily convert our ROS data to PCL data.

#### Statistical Outlier Filtering
When data has a significant amount of noise due to external factors like dust, humidity, or various light sources, it can lead to outliers in the data that can
throw off the results we generate. To counter this, outlier filtering is helpful in reducing the external noise. In this case, we performed a statistical
analysis in the neighborhood of each point, and removed the points which did not meet certain criteria. In this project, we used PCL's
StatisticalOutlierRemoval filter. This filter takes each point in the point cloud and computes the distance to all of its neighbors, and then calculates
a mean distance. By assuming a Gaussian distribution, all points whose mean distances are outside of an interval defined by the global distances
mean+standard_deviation are considered to be outliers and removed from the point cloud.

#### Voxel Grid Downsampling
In order to speed up analysis of point cloud data, voxel grid downsampling was employed. Since we are using RGB-D cameras, we are provided with
very feature rich and dense point clouds. Analyzing the raw point cloud data can lead to slower computations and relatively little improvement
over a more sparsely populated point cloud. This is where voxel (volume element) grid downsampling comes in. A voxel grid filter allows you to downsample
the point cloud from the RGB-D camera by taking a spatial average of the points in the cloud confined by each voxel. This was performed in
this project by the function VoxelGrid provided by the PCL library.

#### Pass Through Filtering
When processing data from an image, there is usually a large amount of the image that isn't necessary for our purposes. It is usually
helpful for us to crop the image, only retaining what is useful for our object recognition purposes. This is where pass through
filtering is handy. This acts as sort of a cropping tool, allowing us to take out any part of the image that isn't
the focus of our object recognition. For this project, it was important for us to retain the table and the objects sitting on the
table, but the rest of the image (the floor, the table leg, etc.) could be cropped out of the image. A pass through filter allowed us
to accomplish this by setting which dimensions of the image we are interested in.

#### RANSAC Plane Segmentation
Random Sample Consensus (RANSAC) was used in this project to help identify points in the dataset that belong to
a particular model. In this project, the models were the different items sitting on the table in front of the
PR2 robot. Essentially, the RANSAC algorithm splits the data in a dataset into either inliers or outliers, where inliers
are defined by a particular model with certain parameters. Outliers are those that don't fit, and can be discarded.
For this project, it is easy to model the table, and then have it removed, leaving only the objects on the table. PCL
provides us with the RANSAC algorithm, which was used in this project to perform plane fitting.

#### Euclidean Clustering
Clustering is used to separate data into segments that are related to each other. In this project, we used the Density-Based Spatial Clustering
of Applications with Noise (DBSCAN) algorithm, which is also known as Euclidean Clustering since the decision of whether to place a point in a cluster
is dependent on the Euclidean distance between that point and other cluster members. PCL provides a function called
EuclideanClusterExtraction that performs a DBSCAN cluster search on our 3D point cloud.

#### Generating Features
For this step, we ran a script to capture and save features for each of the objects used in different test worlds. The capture sessions consisted
of capturing different poses of different objects using a RGB-D camera. For this project, we could set how many different poses each object could be
captured in. In each pick list, 50 poses of each object was captured. This provided me with sufficient accuracy to complete the project. This could
definitely be modified, however I did not want to over train the SVM and end up with memorization over learning.

#### Training the Support Vector Machine (SVM)
Once the different features of the world objects were captured, we proceeded to train our SVM. A SVM is a particular type of supervised
machine learning algorithm that allows us to characterize the parameter space of our dataset into discrete classes. In this case, we utilized
the sklearn package in Python, which provides us with a SVM implementation. For this project, several parameters were set to help with
the classification. First, the 'linear' kernel was used for the SVM. Next, we converted our RGB color data into HSV color data. The reasoning
behind this is that HSV color is much less effected by changes in light intensity. Lastly, a bin strategy on our histogram
data, utilizing 32 bins, was used in this project. Shown below are the confusion matrices for each pick list.

##### Pick List 1
![pick_list_1_svm][image1]

##### Pick List 2
![pick_list_2_svm][image3]

##### Pick List 3
![pick_list_3_svm][image5]

For each of the pictures above, the top matrix is the total count of actual labels vs the predicted label. The bottom matrix is
the normalized count of actual vs predicted.

#### Object Recognition
Once we have a trained SVM, we can use that to locate and identify the objects in our simulation! Included below are images of the
object recognition performed by the PR2 Robot.

##### Pick List 1
![pick_list_1_obj_recog][image2]

##### Pick List 2
![pick_list_2_obj_recog][image4]

##### Pick List 3
![pick_list_3_obj_recog][image6]

As you can see from the picture, the object recognition is quite accurate. It is worth noting that in Pick List 3, the glue
object was not identified. One explanation is that it is located quite close and behind another object. An adjustment of the
parameters used above should help to separate the glue object from the object that it is behind.

#### Outputting to .yaml file
The following performance was seen in the different environments:

test1.world - 100% of the objects were recognized and correctly labeled

test2.world - 100% of the objects were recognized and correctly labeled

test3.world - 87.5% of the objects were recognized and correctly labeled (the exception being the glue)

After the objects were recognized, we needed to output certain parameters to .yaml files. First, we need to make sure that
the data is robust enough to record. For this, conditional statements were used. First, the object_list parameters were
pulled from the ros server. The length of the object_list would tell us which test scene we are in. Once we have identified
and labeled the objects seen, a set of the labels identified was produced. If the set of identified labels was not above a minimum
level (6 for test3, 4 for test2, and 3 for test1), the data was not recorded.

Next, we loop through the objects in our parameter list. If we identified that object on the table, we calculate the centroid of
that object, as well as which group it is a part of, which tells us which arm will be responsible for grabbing that object. That centroid
gives us the parameters for our pick_pose. Lastly, we can grab the parameters for our place_pose from dropbox.yaml. From all of this
data, we have a complete set of parameters to export to the necessary output yaml file and send to the pick and place server. For this project
the pick and place component was not implemented and beyond the acceptance criteria of this project, although I do anticipate on completing this part
of the project because this was quite interesting!