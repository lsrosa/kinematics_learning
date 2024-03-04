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

DEBUG = 1

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
    n = get_njoints(arm)
    fstr = "{:6.3f} " * n
    pstr = fstr.format(*joints_pos) 
    vstr = fstr.format(*joints_vel) 
    estr = fstr.format(*joints_eff) 
    print( f"pos: {pstr} | vel: {vstr} | eff: {estr}" )


# blocking function
def goto_joints_position(pub_send_joints,arm,target,rate,joint_pos_tolerance=0.05):
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
            print_joint_state(arm)

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
                print_joint_state(arm)
            rate.sleep()
            it -= 1





def main():
    global pub_send_joints

    arm = 'r3_arm'
    
    print(f"Init ROS node for {arm} control...")
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
    pub_send_joints = rospy.Publisher(f"/{arm}/joint_group_position_controller/command", Float64MultiArray, queue_size=1)
    sub_state = rospy.Subscriber(f"/{arm}/joint_states", JointState, jointState_cb)

    print("Wait for joint pos...")
    # wait for some joint position
    while joints_pos==None:
        rate.sleep()

    rospy.sleep(1)  # needed to wait for controllers to be active

    joint_pos_tolerance=0.05

    if arm=='r2_arm':    
        targets = [ [0,0], [0.3,-0.4], [-0.3,0.4], [0,0] ]
    elif arm=='r3_arm':    
        targets = [ [0,0,0], [0.6,-0.8,0.6], [-0.6,0.8,-0.6], [0,0,0] ]
    else:
        targets = [ [0,0,0,0], [0.3,-0.4,0.5,-0.6], [-0.3,0.4,-0.5,0.6], [0,0,0,0] ]
    
    for t in targets:
        print(f"Send joints position {t}")
        goto_joints_position(pub_send_joints,arm,t,rate,joint_pos_tolerance)
        rate.sleep()

    sub_state.unregister()
    pub_send_joints.unregister()


if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")



