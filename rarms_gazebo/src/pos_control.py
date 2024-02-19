#!/usr/bin/python
#
# Joint position control

import math
import numpy as np


import rospy
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
import tf.transformations as tft

#from gazebo_models import *

DEBUG = 0

# global
pub_send_joints = None
joints_pos = None

def jointState_cb(data):
    global joints_pos
    joints_pos = data.position


# blocking function
def goto_joints_position(pub_send_joints,target,rate,joint_pos_tolerance=0.05):
    global joints_pos

    # print("Target position: %s" %joint_str(target))

    message = Float64MultiArray()
    message.data = target

    reached = False

    while not rospy.is_shutdown() and not reached:
        pub_send_joints.publish(message)
        reached = True
        d = [0,0,0,0]
        k = 0
        for u,v in zip(joints_pos, target):
            d[k] = v-u
            k+=1
            if abs(u-v)>joint_pos_tolerance:
                reached = False
                if DEBUG==0:
                    break

        if DEBUG>0:
            print("++ pos %s - d %s / %.3f" 
                %(joint_str(joints_pos),joint_str(d),joint_pos_tolerance))

        rate.sleep()

    if reached:
        # wait more
        it = 3
        while not rospy.is_shutdown() and it>0:
            k = 0
            for u,v in zip(joints_pos, target):
                d[k] = v-u
                k+=1
            if DEBUG>0:
                print(".. pos %s - d %s / %.3f" 
                    %(joint_str(joints_pos),joint_str(d),joint_pos_tolerance))
            rate.sleep()
            it -= 1





def main():
    global pub_send_joints

    rospy.init_node('pos_control', anonymous=True)
    rate = rospy.Rate(10) 

    # wait for ROS clock to publish something different from 0 
    # (needed only once)
    t0 = rospy.Time.now()
    while t0.secs==0 and t0.nsecs==0:
        rate.sleep()
        t0 = rospy.Time.now()
    t1 = rospy.Time.now()

    # publishers and subscribers must be created only once
    pub_send_joints = rospy.Publisher('/r4_arm/joints_position_controller/command', Float64MultiArray, queue_size=1)
    sub_state = rospy.Subscriber('/r4_arm/joint_states', JointState, jointState_cb)


    # wait for some joint position
    while joints_pos==None:
        rate.sleep()

    joint_pos_tolerance=0.05

    # init start position
    target = [0,0,0,0]
    goto_joints_position(pub_send_joints,target,rate,joint_pos_tolerance)
    rate.sleep()

    target = [0.3,-0.4,0.5,-0.6]
    goto_joints_position(pub_send_joints,target,rate,joint_pos_tolerance)
    rate.sleep()

    target = [-0.3,0.4,-0.5,0.6]
    goto_joints_position(pub_send_joints,target,rate,joint_pos_tolerance)
    rate.sleep()

    target = [0,0,0,0]
    goto_joints_position(pub_send_joints,target,rate,joint_pos_tolerance)
    rate.sleep()

    sub_state.unregister()
    pub_send_joints.unregister()


if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")
        state_file.close()


