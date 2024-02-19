#!/usr/bin/python
#
# Joint position control

import math, copy
import numpy as np


import rospy
from std_msgs.msg import Float64MultiArray, String
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

def jointState_cb(data):
    global joints_pos, joints_vel, joints_eff
    joints_pos = data.position
    joints_vel = data.velocity
    joints_eff = data.effort

def print_joint_state():
    global joints_pos, joints_vel, joints_eff
    pstr = "{:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(*joints_pos) 
    vstr = "{:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(*joints_vel) 
    estr = "{:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(*joints_eff) 
    print( f"pos: {pstr} | vel: {vstr} | eff: {estr}" )


def getTrajetoryMsg(target):
    now = rospy.Time.now()

    tr = JointTrajectory()
    tr.header.seq = 1
    tr.header.stamp = now
    tr.header.frame_id = ''
    tr.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4']

    tr.points = [ ]    
    ts = rospy.Time(0)
    p1 = JointTrajectoryPoint()

    for t in target:
        p1.positions = t.copy()
        # p1.velocities = []
        # p1.accelerations = []
        # p1.effort = []
        ts = ts + rospy.Duration(1.0)
        p1.time_from_start = ts
        tr.points.append(copy.copy(p1))

    return tr

# blocking function
def goto_joints_traj(act_traj,target,rate):
    global joints_pos, joints_vel, joints_eff

    act_traj.wait_for_server()

    goal = FollowJointTrajectoryGoal()

    goal.trajectory = getTrajetoryMsg(target)

    goal.path_tolerance = []
    goal.goal_tolerance = []
    jt = JointTolerance()
    for i in range(1,5):
        jname = "joint_%d" %i
        jt.name = jname
        jt.position = 0.1
        goal.path_tolerance.append(copy.copy(jt))
        goal.goal_tolerance.append(copy.copy(jt))

    goal.goal_time_tolerance = rospy.Duration(0.5)

    act_traj.send_goal(goal)
    rate.sleep()

    finished = False
    while not finished:
        rate.sleep()
        status = act_traj.get_state()
        result = act_traj.get_result() 
        print_joint_state()
        finished = (status == GoalStatus.SUCCEEDED) or (status == GoalStatus.ABORTED)

    print("status: %d - result: %r" %(status,result))

    '''
    # read joints state after trajectory execution
    for _ in range(10):
        rate.sleep()
        print_joint_state()
    '''

# non-blocking function
def send_joints_traj(pub_send_traj,target,rate):

    tr = getTrajetoryMsg(target)
    pub_send_traj.publish(tr)
    rate.sleep()


def main():
    global pub_send_joints

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
    pub_send_traj = rospy.Publisher('/r4_arm/joint_trajectory_controller/command', JointTrajectory, queue_size=1)

    act_traj = actionlib.SimpleActionClient('/r4_arm/joint_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

    sub_state = rospy.Subscriber('/r4_arm/joint_states', JointState, jointState_cb)


    # wait for some joint position
    while joints_pos==None:
        rate.sleep()

    rospy.sleep(1)  # needed to wait for controllers to be active

    # position trajectory
    target = [ [0,0,0,0], [0.3,-0.4,0.5,-0.6], [-0.3,0.4,-0.5,0.6], [0,0,0,0] ]


    print("Send trajetory (non-blocking)")
    send_joints_traj(pub_send_traj,target,rate)  # non-blocking
    print("Waiting...")
    rospy.sleep(7)
    print("Done")

    print("Send trajetory (blocking)")
    goto_joints_traj(act_traj,target,rate)  # blocking
    print("Done")

    sub_state.unregister()
    pub_send_traj.unregister()


if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")
        state_file.close()


