## Project: Kuka KR210 Pick and Place

In this project, code was written for the Kuka KR210 arm to perform Inverse Kinematics. Given a list of end-effector poses, the joint angles for the Kuka KR210 were calculated.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
[//]: # (Image References)

[image1]: ./rsz_kuka_diagram.jpg
[image2]: ./rsz_kuka_angle.jpg
[image3]: ./rsz_grasping.jpg
[image4]: ./rsz_in_transit.jpg
[image5]: ./rsz_over_container.jpg

## Objective
The overall objective of this project was to code the inverse kinematics of a Kuka KR210 arm in order to get it
to perform a pick and place task. In simulation, a can was placed on a shelf that had 9 possible locations that
the can could rest. The arm was then responsible for picking up the can, and placing it in a container. The code
written would help the arm calculate to joint angles needed for the end effector to effectively travel to the can,
and the place it in the container.

#### Kuka KR210 DH parameters
The first step in solving this problem was constructing the DH parameters table.

|  n  |  alpha  |   a   |   d   |  theta      |
| --- |:-------:|:-----:|:-----:|------------:|
|  0  |    0    |   0   |  -    |     -       |
|  1  |  -90    |  0.35 |  0.75 | theta1      |
|  2  |    0    |  1.25 |  0    | theta2 - 90 |
|  3  |  -90    | -0.054|  0    | theta3      |
|  4  |   90    |   0   |  1.50 | theta4      |
|  5  |  -90    |   0   |  0    | theta5      |
|  6  |    0    |   0   |  0    | theta6      |
|  7  |   -     |    -  | 0.303 | theta7      |

For the above table, the column headers represent the following:

1) alpha - twist angle
2) a - link length
3) d - link offset
4) theta - joint angle

The 'alpha' and 'a' symbols are set by the dimensions of the arm itself. 'd' can be a variable (in the case
of a prismatic joint), but since the Kuka KR210 arm is composed entirely of revolute joints, 'd' is also
set by the arm, and is not variable. Only the 'theta' symbol is variable for this arm. 'alpha', 'a', and 'd'
values found in the chart above were found in the URDF specification for the Kuka arm.

Below is a simplified diagram of the Kuka KR210 arm:

![kuka_diagram][image1]

The total transform between each link is composed of 4 individual transforms: 2 rotations and 2 translations. We can summarize this this
total transformation in code as a simple function:
```python
def dh_transform(alpha, a, d, theta):
  dh = Matrix([[              cos(theta),             -sin(theta),           0,               a],
               [ sin(theta) * cos(alpha), cos(theta) * cos(alpha), -sin(alpha), -sin(alpha) * d],
               [ sin(theta) * sin(alpha), cos(theta) * sin(alpha),  cos(alpha),  cos(alpha) * d],
               [                       0,                       0            0                 1]])

```

### Inverse Kinematics

Inverse Kinematics is used in this project to calculate the joint angles necessary for the Kuka arm to move into a position to pick up the
can, and then move it to the container for deposit. The last three joints of the Kuka arm are revolute, and the 3 three joints axes intersect
at a single point, which gives us a spherical wrist situations. In this case, joint 5 is the intersection point, making it the wrist center (WC).

Since we have a spherical wrist, we can decouple the IK problem into Inverse Position and Inverse Orientation. For the Inverse Position problem, the first
three joints govern the position of the wrist center. The following code shows my calculation of the wrist center:
```python
R_EE = np.matmul(np.matmul(np_matrix.rot_z(yaw), np_matrix.rot_y(pitch)), np_matrix.rot_x(roll))

R_correction = np.matmul(np_matrix.rot_z(180 * (np.pi / 180)), np_matrix.rot_y(-90 * (np.pi / 180)))

R_EE = np.matmul(R_EE, R_correction)

End_Effector = np.array([[px], [py], [pz]])

wc_x = End_Effector[0,0] - 0.303 * R_EE[:,2][0]
wc_y = End_Effector[1,0] - 0.303 * R_EE[:,2][1]
wc_z = End_Effector[2,0] - 0.303 * R_EE[:,2][2]
```
The R_correction variable above is a correction matrix that is needed to compensate for the difference between the URDF and DH reference frames
for the end effector. The 0.303 value is from the DH parameter table above (and is a variable in the code), but I explicitly listed it here for
ease of understanding. The roll, pitch, and yaw variables for the gripper are obtained from the simulation in ROS.

Also note that I am using np.array for the calculations in my code. I will talk about that in a later section.

Now that we have the WC position, we need to calculate theta1, theta2, and theta3. Theta1 is pretty straight forward in that it is calculated using the
following code snippet:
```python
theta_1 = np.arctan2(wc_y, wc_x)
```

Now the difficult part. The calculations of theta2 and theta3 are a bit more complicated. The following diagram was provided to facilitate the understanding of the
calculations:

![theta2/3][image2]

In order to calculate theta2 and theta3, we must find the side lengths of this triangle, use the side lengths and the Law of Cosines
to find the angles, and using this information, finally calculate theta2 and theta3 from this information. The following code accomplishes
this task:
```python
side_1 = 1.501
side_2 = np.sqrt(pow((np.sqrt(wc_x * wc_x + wc_y * wc_y) - 0.35), 2) + pow((wc_z - 0.75), 2))
side_3 = 1.25

angle_1 = np.arccos((side_2 * side_2 + side_3 * side_3 - side_1 * side_1) / (2 * side_2 * side_3))
angle_2 = np.arccos((side_1 * side_1 + side_3 * side_3 - side_2 * side_2) / (2 * side_1 * side_3))
angle_3 = np.arccos((side_1 * side_1 + side_2 * side_2 - side_3 * side_3) / (2 * side_1 * side_2))

theta_2 = np.pi / 2 - angle_1 - np.arctan2(wc_z - 0.75, np.sqrt(wc_x * wc_x + wc_y * wc_y) - 0.35)
theta_3 = np.pi / 2 - (angle_2 + 0.036)
```

After calculating the first three thetas, we need the last three. To accomplish this, we first need to get the transformation R0_3. Using the parameters
from the dh table listed above, substituting in the thetas for the first three angles, and using the dh_transform function defined above, we can calculate
the overall transform to the third joint. From there, we take the inverse of the R0_3 and multiply it by R_EE (defined above) to get R3_6. Using R3_6, we
use the following code to calculate the last three angles:
```python
theta_4 = np.arctan2(R3_6[2,2], -R3_6[0,2])
theta_5 = np.arctan2(np.sqrt(R3_6[0,2] * R3_6[0,2] + R3_6[2,2] * R3_6[2,2]), R3_6[1,2])
theta_6 = np.arctan2(-R3_6[1,1], R3_6[1,0])
```

That's it for the angle calculations! This gives the arm the joint angles that it needs to be in to perform each movement.

## Results and Improvements
I have included a video of the arm performing in simulation. It is in the Kuka_Arm_Results_Final.zip file. For sake of time, the video is sped up
by 5x, and each run is labeled. I ran the simulation 8 times, and in all 8 runs, the run was successful! With that being said, there is plenty of
room for improvement. In particular, runs 4, 5, and 6 were quite inefficient. The arm does find the cylinder and successfully deposits
it into the container, however the angles taken to accomplish it are wild. I believe further restricting the joint angles when there
are multiple solutions could help this issue. I also noticed that during almost every run, the wrist rotates much more than what I
believe is necessary to accomplish the goal. Again, this comes down to fine tuning the angles of the wrist joints.
