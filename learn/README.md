# Run

    python3 learn.py -env ReacherMod6vSR -alg SAC -learn_steps 500000 [-seed <n>]


To speep-up tests:

Environment with a fixed target position

    python3 learn.py -env ReacherMod6vSRFT -alg SAC -learn_steps 300000 [-seed <n>]


Environment with a fixed target position and one DOF arm

    python3 learn.py -alg SAC -env ReacherMod6vSRFTA1 -learn_steps 150000 [-seed <n>]

# Save and load models

The script automatically save and reload the models and the log files in the `runs` folder

To reset, just delete the files

    rm -rf runs/<env>_<alg>_<seed>*

# Play the policy

    python3 learn.py -env ReacherMod6vSR -alg SAC [-seed <n>] --play

