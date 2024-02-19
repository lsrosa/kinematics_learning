#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, PoseStamped
import tf2_ros, tf2_geometry_msgs


class GazeboLinkPose:

  def __init__(self, link_names_in,link_names_out,frames):

    self.link_names_in = link_names_in
    self.link_names_out = link_names_out
    self.target_frames = frames
    self.link_poses_out = [None] * len(self.link_names_out)
    self.pose_pub = [None] * len(self.link_names_out)

    self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) 
        # tf_listener needed to receive messages

    self.source_frame = "world"

    time = rospy.Time(0) # last transform available
    while not self.tf_buffer.can_transform(self.target_frames[0], self.source_frame, time):
        print('waiting for transform')
        rospy.sleep(0.1)
    print('OK transform %s %s' %(self.source_frame, self.target_frames[0]))

    self.states_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.callback)
    for i,n in enumerate(self.link_names_out):
        self.pose_pub[i] = rospy.Publisher("/gazebo/" + n, PoseStamped, queue_size = 10)


  def callback(self, data):
    # which names ara available
    # print(data.name)

    try:
        for (i,n) in enumerate(self.link_names_in):

            ind = data.name.index(n)
            pose = data.pose[ind]
            # latest transform
            ltf = self.tf_buffer.lookup_transform(self.target_frames[i], self.source_frame, rospy.Time(0))

            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = ltf.header.stamp
            pose_stamped.header.frame_id = self.source_frame
            pose_stamped.pose = pose

            self.link_poses_out[i] = \
                tf2_geometry_msgs.do_transform_pose(pose_stamped, ltf)

            # print(self.link_poses_out[i])

    except ValueError as e:
      # print(e)
      pass


  def run(self, publish_rate):
    rate = rospy.Rate(publish_rate)
    while not rospy.is_shutdown():
      for i in range(len(self.link_names_out)):
        if self.link_poses_out[i] != None:
            self.pose_pub[i].publish(self.link_poses_out[i])
      rate.sleep()



if __name__ == '__main__':
  try:
    rospy.init_node('gazebo_link_pose', anonymous=True)

    publish_rate = rospy.get_param('~publish_rate', 10)

    link_names_in = ['r4_arm::EE_tip', 'die::die::link', 'die::die::link']
    link_names_out = ['r4_arm_EE', 'die', 'die_tip']
    frames = ['base_link', 'world', 'EE_tip']

    gp = GazeboLinkPose(link_names_in,link_names_out,frames)

    gp.run(publish_rate)

  except rospy.ROSInterruptException:
    pass

