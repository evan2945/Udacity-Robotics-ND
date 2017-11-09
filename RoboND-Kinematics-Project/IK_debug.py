from sympy import *
from time import time
from mpmath import radians
import numpy as np
import tf

'''
Format of test case is [ [[EE position],[EE orientation as quaternions]],[WC location],[joint angles]]
You can generate additional test cases by setting up your kuka project and running `$ roslaunch kuka_arm forward_kinematics.launch`
From here you can adjust the joint angles to find thetas, use the gripper to extract positions and orientation (in quaternion xyzw) and lastly use link 5
to find the position of the wrist center. These newly generated test cases can be added to the test_cases dictionary.
'''

test_cases = {1:[[[2.16135,-1.42635,1.55109],
                  [0.708611,0.186356,-0.157931,0.661967]],
                  [1.89451,-1.44302,1.69366],
                  [-0.65,0.45,-0.36,0.95,0.79,0.49]],
              2:[[[-0.56754,0.93663,3.0038],
                  [0.62073, 0.48318,0.38759,0.480629]],
                  [-0.638,0.64198,2.9988],
                  [-0.79,-0.11,-2.33,1.94,1.14,-3.68]],
              3:[[[-1.3863,0.02074,0.90986],
                  [0.01735,-0.2179,0.9025,0.371016]],
                  [-1.1669,-0.17989,0.85137],
                  [-2.99,-0.12,0.94,4.06,1.29,-4.12]],
              4:[],
              5:[]}

class NumpyCalc:

  def __init__(self):
    pass

 # Returns a numpy array for an x-axis rotation
  def rot_x(self, x):
    mat_x = np.array([[ 1,         0,          0],
                      [ 0, np.cos(x), -np.sin(x)],
                      [ 0, np.sin(x),  np.cos(x)]])
    return mat_x

 # Returns a numpy array for a y-axis rotation
  def rot_y(self, y):
    mat_y = np.array([[  np.cos(y), 0, np.sin(y)],
                      [          0, 1,         0],
                      [ -np.sin(y), 0, np.cos(y)]])
    return mat_y

  # Returns a numpy array for a z-axis rotation
  def rot_z(self, z):
    mat_z = np.array([[ np.cos(z), -np.sin(z), 0],
                      [ np.sin(z),  np.cos(z), 0],
                      [         0,          0, 1]])
    return mat_z

  # This returns a numpy array for the DH transformations used in later calculations
  def dh_transform(self, alpha, a, d, q):
    dh = np.array([[               np.cos(q),              -np.sin(q),              0,                a],
                   [ np.sin(q)*np.cos(alpha), np.cos(q)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                   [ np.sin(q)*np.sin(alpha), np.cos(q)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                   [                       0,                       0,              0,                1]])
    return dh

def handle_calculate_IK(req):
    #rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

       # Create Modified DH parameters using a simple function that allows the thetas to be passed in as needed
       def dh_params(theta1=0, theta2=0, theta3=0, theta4=0, theta5=0, theta6=0):
         s = {"alpha0":        0, "a0":      0, "d1":  0.75, "q1": theta1,
              "alpha1": -np.pi/2, "a1":   0.35, "d2":     0, "q2": theta2 - np.pi/2,
              "alpha2":        0, "a2":   1.25, "d3":     0, "q3": theta3,
              "alpha3": -np.pi/2, "a3": -0.054, "d4":  1.50, "q4": theta4,
              "alpha4":  np.pi/2, "a4":      0, "d5":     0, "q5": theta5,
              "alpha5": -np.pi/2, "a5":      0, "d6":     0, "q6": theta6,
              "alpha6":        0, "a6":      0, "d7": 0.303, "q7": 0}
         return s

       # Initialize service response
       joint_trajectory_list = []
       for x in xrange(0, len(req.poses)):
         np_matrix = NumpyCalc()
         # IK code starts here
         #joint_trajectory_point = JointTrajectoryPoint()
	 joint_trajectory_point = []

         px = req.poses[x].position.x
         py = req.poses[x].position.y
         pz = req.poses[x].position.z

         (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            [req.poses[x].orientation.x, req.poses[x].orientation.y,
                req.poses[x].orientation.z, req.poses[x].orientation.w])

         #R_EE = np_matrix.rot_z(yaw) * np_matrix.rot_y(pitch) * np_matrix.rot_x(roll)
	 R_EE = np.matmul(np.matmul(np_matrix.rot_z(yaw), np_matrix.rot_y(pitch)), np_matrix.rot_x(roll))
         # Correction needed to correct between UDRF and DH reference frames
         #R_correction = np_matrix.rot_z(np.deg2rad(180)) * np_matrix.rot_y(np.deg2rad(-90))
	 R_correction = np.matmul(np_matrix.rot_z(np.deg2rad(180)), np_matrix.rot_y(np.deg2rad(-90)))
         #R_EE = R_EE * R_correction
	 R_EE = np.matmul(R_EE, R_correction)

         End_Effector = np.array([[px], [py], [pz]])
	 wc = End_Effector - 0.303 * R_EE[:,2]

         # Calculating the wrist center
         wc_x = End_Effector[0,0] - dh_params()["d7"] * R_EE[:,2][0]
         wc_y = End_Effector[1,0] - dh_params()["d7"] * R_EE[:,2][1]
         wc_z = End_Effector[2,0] - dh_params()["d7"] * R_EE[:,2][2]

         # Calculate the first joint angle
         theta1 = np.arctan2(wc_y, wc_x)
         theta1 = np.clip(theta1, -185 * (np.pi / 180), 185 * (np.pi / 185))

         # Calculate the sides needed later for calculating the angles that leads to theta2 and theta3 calculations
         side_1 = 1.501
         side_2 = np.sqrt(pow((np.sqrt(wc_x * wc_x + wc_y * wc_y) - 0.35), 2) + pow((wc_z - 0.75), 2))
         side_3 = 1.25

         # Use the Law of Cosines to calculate the angles needed to find theta2 and theta3
         angle_1 = np.arccos((side_2 * side_2 + side_3 * side_3 - side_1 * side_1) / (2 * side_2 * side_3))
         angle_2 = np.arccos((side_1 * side_1 + side_3 * side_3 - side_2 * side_2) / (2 * side_1 * side_3))
         angle_3 = np.arccos((side_1 * side_1 + side_2 * side_2 - side_3 * side_3) / (2 * side_1 * side_2))

         # Calculate the second and third joint angles
         theta2 = np.pi / 2 - angle_1 - np.arctan2(wc_z - 0.75, np.sqrt(wc_x * wc_x + wc_y * wc_y) - 0.35)
         theta2 = np.clip(theta2, -45 * (np.pi / 180), 85 * (np.pi / 180))

         theta3 = np.pi / 2 - (angle_2 + 0.036)
         theta3 = np.clip(theta3, -210 * (np.pi / 180), 65 * (np.pi / 180))

         alpha0, a0, d1, q1 = dh_params()["alpha0"], dh_params()["a0"], dh_params()["d1"], dh_params(theta1=theta1)["q1"]
         alpha1, a1, d2, q2 = dh_params()["alpha1"], dh_params()["a1"], dh_params()["d2"], dh_params(theta2=theta2)["q2"]
         alpha2, a2, d3, q3 = dh_params()["alpha2"], dh_params()["a2"], dh_params()["d3"], dh_params(theta3=theta3)["q3"]

         R1 = np_matrix.dh_transform(alpha0, a0, d1, q1)
         R2 = np_matrix.dh_transform(alpha1, a1, d2, q2)
         R3 = np_matrix.dh_transform(alpha2, a2, d3, q3)
         R0_3 = np.matmul(np.matmul(R1[0:3,0:3], R2[0:3,0:3]), R3[0:3,0:3])

        # Calculate the tranform for the last three joint angles
         R3_6 = np.matmul(np.transpose(R0_3), R_EE)

         # Last three joint angles
         theta5 = np.arctan2(np.sqrt(R3_6[0,2] * R3_6[0,2] + R3_6[2,2] * R3_6[2,2]), R3_6[1,2])
         # This is to take into account that multiple solutions are possible
         if np.sin(theta5) < 0:
             theta4 = np.arctan2(-R3_6[2,2], R3_6[0,2])
             theta6 = np.arctan2(R3_6[1,1], -R3_6[1,0])
         else:
             theta4 = np.arctan2(R3_6[2,2], -R3_6[0,2])
             theta6 = np.arctan2(-R3_6[1,1], R3_6[1,0])


         # Populate response for the IK request
         # In the next line replace theta1,theta2...,theta6 by your joint angle variables
         joint_trajectory_point = [theta1, theta2, theta3, theta4, theta5, theta6]
         joint_trajectory_list.append(joint_trajectory_point)

         print("\n ***** STILL WORKING **** \n")

       def dh_transform(alpha, a, d, q):
         t = Matrix([[            cos(q),           -sin(q),           0,             a],
                     [ sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                     [ sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                     [                 0,                 0,           0,             1]])
         return t

       #rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
       #return CalculateIKResponse(joint_trajectory_list)
       return [theta1, theta2, theta3, theta4, theta5, theta6, wc_x, wc_y, wc_z]

def test_code(test_case):
    ## Set up code
    ## Do not modify!
    x = 0
    class Position:
        def __init__(self,EE_pos):
            self.x = EE_pos[0]
            self.y = EE_pos[1]
            self.z = EE_pos[2]
    class Orientation:
        def __init__(self,EE_ori):
            self.x = EE_ori[0]
            self.y = EE_ori[1]
            self.z = EE_ori[2]
            self.w = EE_ori[3]

    position = Position(test_case[0][0])
    orientation = Orientation(test_case[0][1])

    class Combine:
        def __init__(self,position,orientation):
            self.position = position
            self.orientation = orientation

    comb = Combine(position,orientation)

    class Pose:
        def __init__(self,comb):
            self.poses = [comb]

    req = Pose(comb)
    start_time = time()
    joints = handle_calculate_IK(req) 
    theta1, theta2, theta3, theta4, theta5, theta6, wc_x, wc_y, wc_z = joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], joints[6], joints[7], joints[8]
    

    ## For error analysis please set the following variables of your WC location and EE location in the format of [x,y,z]
    your_wc = [wc_x, wc_y, wc_z] # <--- Load your calculated WC values in this array
    your_ee = [1,1,1] # <--- Load your calculated end effector value from your forward kinematics
    ########################################################################################

    ## Error analysis
    print ("\nTotal run time to calculate joint angles from pose is %04.4f seconds" % (time()-start_time))

    # Find WC error
    print("HERE")
    if not(sum(your_wc)==3):
      print('here')
      wc_x_e = abs(your_wc[0]-test_case[1][0])
      wc_y_e = abs(your_wc[1]-test_case[1][1])
      wc_z_e = abs(your_wc[2]-test_case[1][2])
      wc_offset = sqrt(wc_x_e**2 + wc_y_e**2 + wc_z_e**2)
      print ("\nWrist error for x position is: %04.8f" % wc_x_e)
      print ("Wrist error for y position is: %04.8f" % wc_y_e)
      print ("Wrist error for z position is: %04.8f" % wc_z_e)
      print ("Overall wrist offset is: %04.8f units" % wc_offset)

      # Find theta errors
      t_1_e = abs(theta1-test_case[2][0])
      t_2_e = abs(theta2-test_case[2][1])
      t_3_e = abs(theta3-test_case[2][2])
      t_4_e = abs(theta4-test_case[2][3])
      t_5_e = abs(theta5-test_case[2][4])
      t_6_e = abs(theta6-test_case[2][5])
      print ("\nTheta 1 error is: %04.8f" % t_1_e)
      print ("Theta 2 error is: %04.8f" % t_2_e)
      print ("Theta 3 error is: %04.8f" % t_3_e)
      print ("Theta 4 error is: %04.8f" % t_4_e)
      print ("Theta 5 error is: %04.8f" % t_5_e)
      print ("Theta 6 error is: %04.8f" % t_6_e)
      print ("\n**These theta errors may not be a correct representation of your code, due to the fact \
              \nthat the arm can have muliple positions. It is best to add your forward kinmeatics to \
              \nconfirm whether your code is working or not**")
      print (" ")

    # Find FK EE error
    if not(sum(your_ee)==3):
      ee_x_e = abs(your_ee[0]-test_case[0][0][0])
      ee_y_e = abs(your_ee[1]-test_case[0][0][1])
      ee_z_e = abs(your_ee[2]-test_case[0][0][2])
      ee_offset = sqrt(ee_x_e**2 + ee_y_e**2 + ee_z_e**2)
      print ("\nEnd effector error for x position is: %04.8f" % ee_x_e)
      print ("End effector error for y position is: %04.8f" % ee_y_e)
      print ("End effector error for z position is: %04.8f" % ee_z_e)
      print ("Overall end effector offset is: %04.8f units \n" % ee_offset)





if __name__ == "__main__":
    # Change test case number for different scenarios
    test_case_number = 1
    print("***********")
    print(test_cases[1][1])
    print("***********")

    test_code(test_cases[test_case_number])
