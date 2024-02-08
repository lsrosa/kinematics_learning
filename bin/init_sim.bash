#!/bin/bash

SESSION=sim

tmux -2 new-session -d -s $SESSION

tmux rename-window -t $SESSION:0 'robot'
tmux send-keys -t $SESSION:0 "roslaunch rlearn_gazebo r4.launch  world:=`rospack find rlearn_gazebo`/worlds/base.world x:=1.302 y:=1.64 z:=0.115 R:=1.571 P:=0 Y:=1.571" 
#C-m

sleep 20

tmux new-window -t $SESSION:1 -n 'rviz'
tmux send-keys -t $SESSION:1 "roscd rlearn_gazebo/config/rviz" C-m
#tmux send-keys -t $SESSION:1 "rosrun rviz rviz -d r4.rviz"
#C-m

tmux new-window -t $SESSION:2 -n 'scripts'
tmux send-keys -t $SESSION:2 "roscd rlearn_gazebo/src" C-m


tmux new-window -t $SESSION:3 -n 'more'
tmux send-keys -t $SESSION:3 "roscd rlearn_gazebo/" C-m


while [ 1 ]; do
  sleep 60
done

