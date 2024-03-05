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


class EffortControlRobot():

    def __init__(self, arm_name):
        self.pub_send_joints = None
        self.joints_pos = None
        self.joints_vel = None
        self.joints_eff = None
        self.arm_name = arm_name
        self.njoints = 0
        self.acc_factor = 1.0
        if self.arm_name=='r2_arm':
            self.njoints = 2
            self.acc_factor = 0.5
        elif self.arm_name=='r3_arm':
            self.njoints = 3
            self.acc_factor = 0.05
        elif self.arm_name=='r4_arm':
            self.njoints = 4
        self.init_ROS()

    def init_ROS(self):
    
        print("Init node...")
        rospy.init_node('traj_control', anonymous=True, disable_signals=True)

        rate = rospy.Rate(10) 

        # wait for ROS clock to publish something different from 0 
        # (needed only once)
        t0 = rospy.Time.now()
        while t0.secs==0 and t0.nsecs==0:
            rate.sleep()
            t0 = rospy.Time.now()
        t1 = rospy.Time.now()

        # publishers and subscribers must be created only once
        cmd_topic = f"/{self.arm_name}/joint_group_effort_controller/command"
        state_topic = f"/{self.arm_name}/joint_states"
        
        print("Subscribers...")

        self.pub_send_eff = rospy.Publisher(cmd_topic, Float64MultiArray, queue_size=1)
        self.sub_state = rospy.Subscriber(state_topic, JointState, self.jointState_cb)

        # wait for some joint position
        print("Wait for joints pos...")
        while self.joints_pos==None:
            rate.sleep()


    def __del__(self):
        self.sub_state.unregister()
        self.pub_send_eff.unregister()
    
    
    def jointState_cb(self, data):
        self.joints_pos = data.position
        self.joints_vel = data.velocity
        self.joints_eff = data.effort


    def print_joint_state(self):
        if self.joints_pos is not None:
            fstr = "{:6.3f} " * self.njoints
            pstr = fstr.format(*self.joints_pos) 
            vstr = fstr.format(*self.joints_vel) 
            estr = fstr.format(*self.joints_eff) 
            rospy.loginfo( f"pos: {pstr} | vel: {vstr} | eff: {estr}" )


    def effort_sample(self):
        r = []
        for i in range(self.njoints):
            e = (np.random.random()*2-1) * self.acc_factor
            r.append(e)  # [-1,1)  * self.acc_factor
        return r

    def stop(self):
        r = []
        for i in range(self.njoints):
            r.append(0)
        self.send_control(r,1)


    # blocking (duration)
    def send_control(self, target, duration):
        print(f"send control {target} for {duration} s")
        duration = rospy.Duration(duration)
        msg = Float64MultiArray()
        msg.data = target
        rate = rospy.Rate(10)
        t0 = rospy.Time.now()
        t1 = t0
        while (t1-t0<duration):
            self.pub_send_eff.publish(msg)
            rate.sleep()
            t1 = rospy.Time.now()





def effort_test(arm, cmd):

    r = EffortControlRobot(arm)

    try:
        if cmd=='stop':
            pass
        elif cmd=='sample':
            t = r.effort_sample()
            r.send_control(t,3)
        elif cmd=='random':
            while True:
                t = r.effort_sample()
                r.send_control(t,1)
        else:
            if arm=='r2_arm':    
                targets = [ ([0,0],1),  ([0.3,-0.4],3), ([-0.3,0.4],3), ([0,0],1) ]
            elif arm=='r3_arm':    
                targets = [ ([0.002,0,0],1), ([0.003,-0.005,0.003],3), ([-0.005,0.005,-0.005],3), ([0,0,0],1) ]

            for t in targets:
                r.send_control(t[0],t[1])
        
    except KeyboardInterrupt:
        print ("User quit!")

    r.stop()






if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('arm', type=str, help='Arm name (r2_arm)')

    parser.add_argument('cmd', type=str, help='Command (random)')

    args = parser.parse_args()
    
    effort_test(args.arm, args.cmd)



