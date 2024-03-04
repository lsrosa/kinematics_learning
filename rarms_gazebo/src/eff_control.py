#!/usr/bin/python
#
# Joint position control

import math, copy
import numpy as np
import argparse

import rospy
from std_msgs.msg import Float64MultiArray, String, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance

import tf.transformations as tft

import actionlib
from actionlib_msgs.msg import *



DEBUG = 0

# global
pub_send_joints = None
joints_pos = None
joints_vel = None
joints_eff = None

def get_njoints(arm):
    n = 0
    if arm=='r2_arm':
        n = 2
    elif arm=='r3_arm':
        n = 3
    elif arm=='r4_arm':
        n = 4
    return n

def jointState_cb(data):
    global joints_pos, joints_vel, joints_eff
    joints_pos = data.position
    joints_vel = data.velocity
    joints_eff = data.effort

def print_joint_state(arm):
    global joints_pos, joints_vel, joints_eff
    if joints_pos is not None:
        n = get_njoints(arm)
        fstr = "{:6.3f} " * n
        pstr = fstr.format(*joints_pos) 
        vstr = fstr.format(*joints_vel) 
        estr = fstr.format(*joints_eff) 
        rospy.loginfo( f"pos: {pstr} | vel: {vstr} | eff: {estr}" )





def effort_test(arm):

    global pub_send_eff
    
    rate = rospy.Rate(25) 

    msg = Float64MultiArray()


    if arm=='r2_arm':    
        targets = [ [0,0], [0.3,-0.4], [-0.3,0.4], [0,0] ]
    elif arm=='r3_arm':    
        targets = [ [0.002,0,0], [0.003,-0.005,0.003], [-0.005,0.005,-0.005], [0,0,0] ]

    msg.layout = MultiArrayLayout()

    for t in targets:
        msg.data = t
        for i in range(200):
            pub_send_eff.publish(msg)
            #print(msg)
            print_joint_state(arm)
            rate.sleep()




if __name__ == '__main__':

    arm = 'r3_arm' # 'r4_arm'

    print("Init node...")
    rospy.init_node('traj_control', anonymous=True)

    rate = rospy.Rate(10) 

    # wait for ROS clock to publish something different from 0 
    # (needed only once)
    t0 = rospy.Time.now()
    while t0.secs==0 and t0.nsecs==0:
        rate.sleep()
        t0 = rospy.Time.now()
    t1 = rospy.Time.now()

    # publishers and subscribers must be created only once
    cmd_topic = f"/{arm}/joint_group_effort_controller/command"
    state_topic = f"/{arm}/joint_states"
    
    print("Subscribers...")

    pub_send_eff = rospy.Publisher(cmd_topic, Float64MultiArray, queue_size=1)
    sub_state = rospy.Subscriber(state_topic, JointState, jointState_cb)

    # wait for some joint position
    print("Wait for joints pos...")
    while joints_pos==None:
        rate.sleep()

    try:
        effort_test(arm)
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")

    sub_state.unregister()
    pub_send_eff.unregister()


