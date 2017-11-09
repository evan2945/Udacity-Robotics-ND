#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
import scipy
from scipy import linalg
import numpy as np

'''
I used Numpy for this project instead of Sympy! I found the amount of time it took Sympy to calculate the inverse kinematics was way too long.
Numpy performs the calculations much faster.
'''

# Class for performing numpy array manipulations. A class is probably overkill; this could probably just be a few functions, but I thought I was going to do more with this.
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
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
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
            joint_trajectory_point = JointTrajectoryPoint()

            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

	    #R_EE = np_matrix.rot_z(yaw) * np_matrix.rot_y(pitch) * np_matrix.rot_x(roll) 
	    R_EE = np.matmul(np.matmul(np_matrix.rot_z(yaw), np_matrix.rot_y(pitch)), np_matrix.rot_x(roll))

	    # Correction needed to correct between UDRF and DH reference frames
	    #R_correction = np_matrix.rot_z(180 * (np.pi / 180)) * np_matrix.rot_y(-90 * (np.pi / 180))
	    R_correction = np.matmul(np_matrix.rot_z(np.deg2rad(180)), np_matrix.rot_y(np.deg2rad(-90)))

	    #R_EE = R_EE * R_correction
	    R_EE = np.matmul(R_EE, R_correction)

	    End_Effector = np.array([[px], [py], [pz]])

	    # Calculating the wrist center
	    wc_x = End_Effector[0,0] - dh_params()["d7"] * R_EE[:,2][0]
	    wc_y = End_Effector[1,0] - dh_params()["d7"] * R_EE[:,2][1]
	    wc_z = End_Effector[2,0] - dh_params()["d7"] * R_EE[:,2][2]

	    # Calculate the first joint angle
	    theta_1 = np.arctan2(wc_y, wc_x)
	    theta_1 = np.clip(theta_1, np.deg2rad(-190), np.deg2rad(190))

	    # Calculate the sides needed later for calculating the angles that leads to theta2 and theta3 calculations
	    side_1 = 1.501
	    side_2 = np.sqrt(pow((np.sqrt(wc_x * wc_x + wc_y * wc_y) - 0.35), 2) + pow((wc_z - 0.75), 2))
	    side_3 = 1.25

 	    # Use the Law of Cosines to calculate the angles needed to find theta2 and theta3
	    angle_1 = np.arccos((side_2 * side_2 + side_3 * side_3 - side_1 * side_1) / (2 * side_2 * side_3))
	    angle_2 = np.arccos((side_1 * side_1 + side_3 * side_3 - side_2 * side_2) / (2 * side_1 * side_3))
	    angle_3 = np.arccos((side_1 * side_1 + side_2 * side_2 - side_3 * side_3) / (2 * side_1 * side_2))
	    
	    # Calculate the second and third joint angles
	    theta_2 = np.pi / 2 - angle_1 - np.arctan2(wc_z - 0.75, np.sqrt(wc_x * wc_x + wc_y * wc_y) - 0.35)
	    theta_2 = np.clip(theta_2, np.deg2rad(-45), np.deg2rad(90))

	    theta_3 = np.pi / 2 - (angle_2 + 0.036)
	    theta_3 = np.clip(theta_3, np.deg2rad(-210), np.deg2rad(70))

	    alpha0, a0, d1, q1 = dh_params()["alpha0"], dh_params()["a0"], dh_params()["d1"], dh_params(theta1=theta_1)["q1"]
	    alpha1, a1, d2, q2 = dh_params()["alpha1"], dh_params()["a1"], dh_params()["d2"], dh_params(theta2=theta_2)["q2"]
	    alpha2, a2, d3, q3 = dh_params()["alpha2"], dh_params()["a2"], dh_params()["d3"], dh_params(theta3=theta_3)["q3"]

	    R1 = np_matrix.dh_transform(alpha0, a0, d1, q1) 
	    R2 = np_matrix.dh_transform(alpha1, a1, d2, q2) 
	    R3 = np_matrix.dh_transform(alpha2, a2, d3, q3) 
	    #R0_3 = R1[0:3,0:3] * R2[0:3,0:3] * R3[0:3,0:3]
	    R0_3 = np.matmul(np.matmul(R1[0:3,0:3], R2[0:3,0:3]), R3[0:3,0:3])

	   # Calculate the tranform for the last three joint angles
	    # R3_6 = np.linalg.inv(R0_3) * R_EE
	    #R3_6 = np.transpose(R0_3) * R_EE
	    R3_6 = np.matmul(np.transpose(R0_3), R_EE)

	    # Last three joint angles
	    theta_5 = np.arctan2(np.sqrt(R3_6[0,2] * R3_6[0,2] + R3_6[2,2] * R3_6[2,2]), R3_6[1,2])
	    if np.sin(theta_5) < 0:
	      theta_4 = np.arctan2(-R3_6[2,2], R3_6[0,2])
	      theta_6 = np.arctan2(R3_6[1,1], -R3_6[1,0])
	    else:
	      theta_4 = np.arctan2(R3_6[2,2], -R3_6[0,2])
	      theta_6 = np.arctan2(-R3_6[1,1], R3_6[1,0])

	    
            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
	    joint_trajectory_point.positions = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]
	    joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)

def dh_transform(alpha, a, d, q):
    t = Matrix([[            cos(q),           -sin(q),           0,             a],
		[ sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
		[ sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
		[                 0,                 0,           0,             1]])
    return t


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
