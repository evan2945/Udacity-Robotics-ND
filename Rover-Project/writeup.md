## Project: Search and Sample Return

In this project, a rover was coded to autonomously map a simulated environment and search for samples of interest.
------------------------------------------------------------------------------------------------------------------
[//]: # (Image References)

[image1]: ./calibration_images/example_grid1.jpg
[image2]: ./output/warped_example.jpg
[image3]: ./output/warped_threshed.jpg
[image4]: ./calibration_images/example_rock1.jpg
[image5]: ./output/threshed_rock.jpg
[image6]: ./output/arrow_map.jpg

## Notebook Analysis

#### Perspective Transform
The first step undertaken in the notebook was the perspective transform. This converts an image from the rover's
point of view to a top-down view of the camera image. The main weight lifting here is performed by openCV:

```python
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped
```

For this to work, we need to determine source and destination points. The source points were determined by overlaying a grid
on the camera image from the rover. This gives us the following image:

![rover_camera_image][image1]

We can use the corners of the closest squares to determine the source points. Through trial and error, appropriate
destination points were discovered, leading us to the following (with dst_size = 5 and bottom_offset=6):
```python
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
```

By applying these source and destination points to the image above, we produce the following perspective transform:

![warped_perspective][image2]

This gives us our top-down perspective.

#### Color Threshold
In this section, we deal with taking the perspective transform and running it through color thresholds, which will help
the rover to differentiate between navigable terrain, obstacles, and rock samples.
The beef of the code comes from the following functions:
```python
def color_thresh(img, rgb_thresh=(160, 160, 160), rgb_thresh_max=(255, 255, 255)):
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2]) \
                & (img[:,:,0] < rgb_thresh_max[0]) \
                & (img[:,:,1] < rgb_thresh_max[1]) \
                & (img[:,:,2] < rgb_thresh_max[2])
    color_select[above_thresh] = 1
    return color_select
```
This takes an image along with two optional arguments that detail rgb thresholds. To differentiate between navigable
terrain and obstacles, an image is passed in and uses the default arguments on rgb_thresh and rgb_thresh_max. If we take the perspective
transform from above and run it through the color_thresh function, we get the following image:

![warped_threshed][image3]

This gives the rover a clearer picture of what terrain it can navigate. Another color_thresh that I performed in this part was on
the rock. The original picture looks like this:

![rock][image4]

The following code was passed in to perform the color thresh:
```python
threshed_rock = color_thresh(rock_img, rgb_thresh=(110, 90, 0), rgb_thresh_max=(255, 200, 50))
```
This produced the following result:

![threshed_rock][image5]

This is definitely something that could use a tiny bit of refinement through the rgb thresholds passed in, however, I have found
that this works sufficiently well during the project.

#### Coordinate Transformations
This section deals with converting our rover's coordinates to several useful forms. I will not paste all of the code here, as there is
quite a bit, but you can find the pertinent parts in the perception.py script. The following steps were performed
1) Convert the pixels of navigable terrain in front of the rover to rover-centric coordinates, where the camera of the rover is at (x, y) = (0, 0).
2) Convert the rover-centric coordinates to world coordinates. This is handled in two steps: rotation and then translation.
3) Convert the rover-centric coordinate to polar coordinates, which will give us the distances and angles to navigable terrain, obstacles,
and rock samples.


#### Process Image
In this section, we apply the steps we have outlined above to produce a video showcasing the different functions. The only additional step here
is to update our world map to reflect what the rover is reading in. This is done by the following code:
```python
    data.worldmap[y_world_obst, x_world_obst, 0] += 255
    data.worldmap[y_world_rock, x_world_rock, 1] += 255
    data.worldmap[y_world, x_world, 2] += 255
```
The one thing to note here is that during actual navigation, we want to be careful of the roll and pitch angles to preserve fidelity.
After all of this has been executed, we can now create a mp4 of our data. This has been produced already and lives in the output
folder, labeled as test_mapping.mp4. Take a look!


## Autonomous Navigation and Mapping

#### Perception Step
The perception step follows many of the same steps outlined in the Notebook section above. As a brief recap, the following actions were taken:
1) Source and destinations were chosen in order to perform the perspective transform. These were the same as noted above in the Navigation section.
2) Perform the perspective transform.
3) Perform the color thresholds on navigable terrain, obstacles, and rock samples
4) Convert the rover-centric coordinates to world coordinates.
5) Convert the navigable terrain rover coordinates to polar coordinates, obtaining distances and angles.
6) Convert the rock sample rover coordinates to polar coordinates, obtaining distances and angles to samples (if present).

A few more steps were performed here in order to increase the performance of the mapping and sample collection.
First, I only recorded the data that the rover saw to the worldmap if the roll and pitch angles were within a certain range.
```python
    if (Rover.pitch < 0.5 or Rover.pitch > 359.5) and (Rover.roll < 0.5 or Rover.roll > 359.5):
        Rover.worldmap[y_world_obst, x_world_obst, 0] += 255
        Rover.worldmap[y_world_rock, x_world_rock, 1] += 255
        Rover.worldmap[y_world_nav, x_world_nav, 2] += 255
```
This really helped in the mapping fidelity as we don't write the data when the rover is bouncing around.

Next, I changed the Rover.nav_dists and Rover.nav_angles depending if we were near a rock sample or not.
```python
    if len(rock_dist) > 0 and np.mean(rock_dist) < 60:
        Rover.mode = 'rock_visible'
        Rover.nav_dists = rock_dist
        Rover.nav_angles = rock_angle
```
In the case where we can see a rock sample, and it is less than a certain distance away, we should switch to rock_visible mode
and set the nav_dists and nav_angles to the rock polar coordinates so we can navigate to it for pickup. The distance criteria may
seem odd, and it will be mentioned later, but the short story is that I want my rover to hug the left wall and I don't want it to
leave the left wall and travel to the right side to pick up a sample as it will get to it eventually.

#### Decision Step
I will step through the code that I added to this file (I won't really explain the code that was given to us).
##### Rock Visible mode
```python
    elif Rover.mode == 'rock_visible':
        mean_dist = np.mean(Rover.nav_dists)
        if (mean_dist < Rover.stop_forward) and (mean_dist > 15) and Rover.vel != 0:
            Rover.throttle = 0.1
        elif mean_dist < 15 and Rover.near_sample:
            Rover.brake = Rover.brake_set
        elif mean_dist < 15 and Rover.brake > 5 and Rover.vel == 0 and not Rover.near_sample:
            Rover.brake = 0
            Rover.throttle = 0.1
        else:
            Rover.throttle = Rover.throttle_set
        # Set steering to average angle clipped to the range +/- 15
        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
```
This handles the situation in which a rock sample is in view. As we approach the sample, we want to ease off of the throttle
and essentially coast to the rock. Once at the rock, we stop completely to allow the rover to pick up the sample. After
the sample has been collected, we set our throttle and continue with the mapping.

##### Turning mode
```python
    elif Rover.mode == 'turning':
            Rover.correcting_loop += 1
            Rover.brake = 0
            Rover.throttle = 0
            Rover.steer = -15
            if Rover.vel > 0.5:
                if Rover.correcting_loop > 50:
                    Rover.mode = 'forward'
                    Rover.correcting_loop = 0
            else:
                if Rover.correcting_loop > 100:
                    Rover.mode = 'forward'
                    Rover.correcting_loop = 0
```
This mode keys off of a counter that indicates that we have been turning for too long, essentially caught in a circular loop.
In order to get out of this situation, we steer in the opposite direction for a set amount of readings. For this I chose to determine
how long to counter steer based on how fast the rover was moving. The main situations that this mode gets called in is
in the beginning when the rover is just starting out. This helps the rover to find a wall so it can wall hug.

##### Stuck mode
```python
elif Rover.mode == 'stuck':
            Rover.throttle = 0
            Rover.steer = -15
            Rover.is_stuck += 1
            if Rover.is_stuck > 125:
                Rover.is_stuck = 0
                Rover.mode = 'forward'
```
This mode keys off of recording that the rover hasn't exceeded a velocity threshold for a certain number of readings.
In this case, we simply adjust the angle that the rover is facing, and switch back to forward mode. If we are then able
to move forward, we go on our way. If we turn into another stuck position, we turn again, until we find a path forward.

##### Wall hugging
```python
    Rover.nav_angles = np.sort(Rover.nav_angles)[-int(len(Rover.nav_angles) * 0.5):]

    if np.max(Rover.nav_angles * 180/np.pi) < 25:
                    mean_angle = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
                # If we are on a straight away, we don't want to consider angles less than 0, as that will induce swaying.
                # This may not be the best solution, but I found it to be effective in simulation.
                else:
                    mean_angle = np.clip(np.mean(Rover.nav_angles * 180/np.pi), 0, 15)
                # Set steer angle
                Rover.steer = mean_angle
```
This was my attempt to hug the left wall. This is favorable as it would allow us to visit as much of the map as possible, as
well as collect as many samples as possible since all samples were located near a wall. The first step is to sort the nav_angles
we set in the perception step. This is done to get the second half of the angles, which will get us the angles closest to the left
wall. I had originally found the mean angle of this set of angles, but noticed a few problems. One of the biggest issues was the
fact that there was a good bit of sway once the rover got to a location that offered wide open spaces to the right. This is problematic
as this causes us to not map the area as our angles are outside of our recording thresholds. The solution that I came up with (through
 trial and error) was to clip the angle that the the rover could turn to be between 0 and 15 degrees. While this works in straight
 aways, the rover wouldn't perform at dead ends. I added the conditional that looks at what the max angle in nav_angles is, and if its
 under a threshold (25 degrees in this case), it would recognize that it is at or near a dead end and allow the rover to turn between
 -15 and 15 degrees. This was satisfactory for me as it stayed quite straight while traveling ahead and made appropriate turns at dead ends.
 There is definitely room for improvement here. For one, sometime the rover comes too close to the wall, which causes it to hit bumps that
 throws off the measurements. Also, there is a particular outcrop of rocks on one particular side of the start that the rover sticks too close to
 and always gets stuck on (it eventually corrects itself, but I sometimes miss the samples located there).

##### Rover variables added by me
```python
    self.is_stuck = 0 # This will help us out if we are stuck for some length of time
    self.turning = 0 # This will hold our turning threshold
    self.correcting_loop = 0 # This will hold our threshold for the correcting turn
    self.start = None # This records our starting position
```

### Results
   For the autonomous navigation, I used the settings of '1024x768' and 'Good' quality. I found through numerous simulations that I can get a mapped percent of over 60%.
If left for some time, it can map over 80%. The fidelity at the lowest was recorded at 65%. On good runs, I recorded fidelity in the low 80's. The rover does a decent
job of finding and collecting the samples. On most runs, I can collect at least 3 samples. The rover sometimes has issues when the sample is behind a sharp
turn (there is an outcropping of two black rocks at the beginning that confuses my rover sometimes), or if the sample is located near the middle of a wide
dead end. More specific guidance for the sample collection code would be useful in correcting these issues. Another issue that I had, which was mentioned above, was significant
sway in the rover when it was traveling down a lane or any place that had significant navigable angles on both sides. This would throw off the rover's mapping attempts as the
data wasn't good enough to record. This was fixed by preventing the angles that the rover could turn based on what angles are available. A further modification was needed
as this wasn't a good solution in cases where we ran into dead ends (see Wall hugging section for more information). Another issue with navigation (that wasn't addressed) was the
fact that during the wall hugging, sometimes the rover strays to close to one side, causing it to hit the bumps near the wall. This throws off our mapping, which is unfavorable.
It also make the rover susceptible to getting stuck near rock outcroppings.

One of the improvements that could be made is the speed of the rover and the time it takes to navigate. It is possible for the rover to explore 90+% of the map, however, it takes
a decent amount of time. Changing the max speed at which the rover can travel would alleviate that, but more code would be needed to manage small corrections in the rover's path, as increased
speed would cause small mistakes to be amplified. This improvement was not attempted in this project, but could be a fun addition to its functionality. I did not attempt to tackle the part of the
challenge in which you return to the starting point once all of the samples are collected. By keeping up with the initial starting position (which I actually include in perception), it would be
possible to implement this functionality for the completeness of the challenge. The rover never consistently collected all 6 samples, so I didn't get around to completing this part of
the challenge, although this would be a great cherry on top!