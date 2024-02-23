#!/bin/bash

SESSION=sim

tmux -2 new-session -d -s $SESSION

tmux rename-window -t $SESSION:0 'robot'
tmux send-keys -t $SESSION:0 "roslaunch rarms_gazebo r4.launch  world:=`rospack find rarms_gazebo`/worlds/base.world x:=1.302 y:=1.64 z:=0.115 R:=1.571 P:=0 Y:=1.571" 
#C-m

tmux new-window -t $SESSION:1 -n 'make'
tmux send-keys -t $SESSION:1 "cd ~/ros/catkin_ws/src" C-m
tmux send-keys -t $SESSION:1 "ln -s ~/src/iros2024/rarms_gazebo ." C-m
tmux send-keys -t $SESSION:1 "cd .." C-m
tmux send-keys -t $SESSION:1 "catkin_make" C-m

sleep 20

tmux new-window -t $SESSION:2 -n 'rviz'
tmux send-keys -t $SESSION:2 "roscd rarms_gazebo/config/rviz" C-m
#tmux send-keys -t $SESSION:2 "rosrun rviz rviz -d r4.rviz"
#C-m

tmux new-window -t $SESSION:3 -n 'scripts'
tmux send-keys -t $SESSION:3 "roscd rarms_gazebo/src" C-m





while [ 1 ]; do
  sleep 60
done

