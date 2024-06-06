#!/bin/bash

SESSION=learn

tmux -2 new-session -d -s $SESSION


tmux rename-window -t $SESSION:0 'tb'
tmux send-keys -t $SESSION:0 "cd ../learn" C-m
tmux send-keys -t $SESSION:0 "tensorboard --logdir=runs" C-m

tmux new-window -t $SESSION:1 -n '1'
tmux send-keys -t $SESSION:1 "cd ../learn" C-m

tmux new-window -t $SESSION:2 -n '2'
tmux send-keys -t $SESSION:2 "cd ../learn" C-m

tmux new-window -t $SESSION:3 -n '3'
tmux send-keys -t $SESSION:3 "cd ../learn" C-m


while [ 1 ]; do
  sleep 60
done

