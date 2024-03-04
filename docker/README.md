# Install

## CPU

    ./build_gazebo11.bash
    ./build_ml.bash

## GPU

    TODO, probably not needed for these experiments


# Run

    Run the container


        ./run.bash [X11|nvidia] [service=<learn>gazebo>, default all]

    Enter the container

        docker exec -it <container> tmux a

    e.g.,

        docker exec -it learn tmux a

# Patch

        cp ~/src/iros2024/docker/patch/mujoco_rendering.py ~/.local/lib/python3.8/site-packages/gymnasium/envs/mujoco/

