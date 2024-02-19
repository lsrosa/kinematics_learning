/*
 * Copyright (C) 2012 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/


/*
 *  Inspired by https://classic.gazebosim.org/tutorials?tut=topics_subscribed&cat=transport
 *  and https://github.com/wonwon0/gazebo_contact_republisher
 *
**/

#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>
#include "std_msgs/String.h"

#include <ros/ros.h>

#include <iostream>
#include <string>

// global
ros::Publisher pub;
unsigned int msgid=0;

std::string GZ_CONTACT_TOPIC_NAME = "/gazebo/pglab_base/physics/contacts";  
    // "/gazebo/default/robot/contacts";

/////////////////////////////////////////////////
// Function is called everytime a message is received.
void contacts_cb(ConstContactsPtr &_msg)
{
  // Dump the message contents to stdout.
  // std::cout << _msg->DebugString();

  ros::Time t = ros::Time::now();

  double ts = t.sec + t.nsec/1e9;

  if (_msg->contact_size()==0) {
    std_msgs::String smsg;
    std::stringstream ss;
    ss << ts << ";None;None";
    smsg.data = ss.str();
    pub.publish(smsg);
  }

  // Publish String message with collision names
  for (int i = 0; i < _msg->contact_size(); ++i) {
    std::string c1 = _msg->contact(i).collision1();
    std::string c2 = _msg->contact(i).collision2();
    std_msgs::String smsg;
    std::stringstream ss;
    ss << ts << ";" << c1 << ";" << c2;
    smsg.data = ss.str();
    pub.publish(smsg);
  }

#if 0

    // Publish contacts message with all information
    gazebo_contacts::contacts_msg contacts_message;

    // TODO check
    std::Header header;
    header.seq = msgid++;
    header.stamp = ros::Time::now();
    header.frame = "";

    std::vector<gazebo_contacts::contact_msg> contacts_list;

    for (int i = 0; i < _msg->contact_size(); ++i) {
        contact_republisher::contact_msg contact_message;

        contact_message.collision_1 = _msg->contact(i).collision1();
        contact_message.collision_2 = _msg->contact(i).collision2();

        contact_message.normal[0] = _msg->contact(i).normal().Get(0).x();
        contact_message.normal[1] = _msg->contact(i).normal().Get(0).y();
        contact_message.normal[2] = _msg->contact(i).normal().Get(0).z();

        contact_message.position[0] = _msg->contact(i).position().Get(0).x();
        contact_message.position[1] = _msg->contact(i).position().Get(0).y();
        contact_message.position[2] = _msg->contact(i).position().Get(0).z();

        contact_message.depth = _msg->contact(i).depth().Get(0);

        contacts_list.push_back(contact_message);
    }

    /*
    // ??? is this needed ???
    if ( _msg->contact_size() == 0){
        contact_republisher::contact_msg contact_message;

        contact_message.collision_1 = "";
        contact_message.collision_2 = "";

        contact_message.normal[0] = 0;
        contact_message.normal[1] = 0;
        contact_message.normal[2] = 0;

        contact_message.position[0] = 0;
        contact_message.position[1] = 0;
        contact_message.position[2] = 0;

        contact_message.depth = 0;

        contacts_list.push_back(contact_message);
    }
    */

    contacts_message.header = header;
    contacts_message.contacts = contacts_list;
    pub.publish(contacts_message);


#endif

}

/////////////////////////////////////////////////
int main(int _argc, char **_argv)
{
  // Load gazebo
  gazebo::client::setup(_argc, _argv);
  ros::init(_argc, _argv, "gazebo_contacts");
  ros::NodeHandle nh;
  pub = nh.advertise<std_msgs::String>("contacts", 10);

  // Create our node for communication
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init();

  // Listen to Gazebo contacts topic
  gazebo::transport::SubscriberPtr sub = node->Subscribe(GZ_CONTACT_TOPIC_NAME, contacts_cb);

  // main loop
  float loop_freq = 10.0; // Hz
  while (ros::ok()) {
    gazebo::common::Time::MSleep(1000.0/loop_freq);
    ros::spinOnce();
  }

  // Make sure to shut everything down.
  gazebo::client::shutdown();
}

