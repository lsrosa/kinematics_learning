# Run

        python3 learn.py -env ReacherMod6vSR -alg SAC -learn_steps 700000 [-seed <n>]


To speep-up tests:

Environment with one DOF arm

        python3 learn.py -env ReacherMod6vSRA1 -alg SAC -learn_steps 200000 [-seed <n>]

Environment with fixed target position

        python3 learn.py -env ReacherMod6vSRFT -alg SAC -learn_steps 250000 [-seed <n>]

Environment with one DOF arm and fixed target position

        python3 learn.py -env ReacherMod6vSRFTA1 -alg SAC -learn_steps 150000 [-seed <n>]


# Monitor learning progresses

Open a browser at

    http://localhost:6006/


# Save and load models

The script automatically save and reload the models and the log files in the `runs` folder

To reset, just delete the files

        rm -rf runs/<env>_<alg>_<seed>*



# Play the policy

        python3 learn.py -env ReacherMod6vSR -alg SAC [-seed <n>] --play

Press 'h' to remove debug text from the simulator window



If you get error when running mujoco simulator, apply this patch
in the docker container

        cp ~/src/iros2024/docker/patch/mujoco_rendering.py ~/.local/lib/python3.8/site-packages/gymnasium/envs/mujoco/


    


# Train and use FKNet

Train FKNet on AbsReacher6v and use it to control the robot on ReacherMod.
No learning on Reader env.

        python3 fknet_learn.py
    
It saves the model parameters in `fknet_AbsReacher.th`

Test the trained network

        python3 fknet_learn.py --test --render

It will compute the difference between the FKNet estimated end effector (EE) position and the real EE position from Reacher env

The values of the errors are small (~ 0.01). Note this error is justified by the
different model of Reacher wrt AbsReacher. 


        python3 fknet_learn.py -eval 100

Evaluation of control in Reacher Env using FKNet



# Learn policy

Learn SAC policy using fknet instead of env FK

        python3 learn.py -env ReacherMod6vSR -alg SAC -learn_steps 700000 [-seed <n>] -fknet fknet_AbsReacher.pth


Compare results obtained with SAC using env FK. They are quite similar.

It shows that estimating FK in an abstract simulator, allows to learn in the concrete simulator, without the perfect knowledge of the FK model in the more concrete simulator.


