- tune_fkine.py - trains fkine with different hyperparams (grid-search)
- compare_link_mono.py - given fkine hyperparams, train models using the Linked and Monolithic architectures 
- tune_policy.py - Search for policy hyperparams using stable-baselines3 zoo (check installation instructions bellow). The policy uses the fkine, which is integrated withing the features_extractor.
- learn_policy_fkine.py - learn policy given policy hyperparams

## Setup

Prepare python venv. Do not forget to activate the venv in every shell.
```sh
git clone git@github.com:lsrosa/model-free-rl.git
cd model-free-rl
python -m venv mfrl
source mfrl/bin/activate
pip install -r requirements.txt
cd ..
```

```sh
git clone git@github.com:lsrosa/kinematics_learning.git
```

## Installing Zoo
We modified Zoo, so checkout our version. 

Create symlinks to this repo:
```sh
cd ../..
git clone git@github.com:lsrosa/rl-baselines3-zoo.git
cd rl-baselines3-zoo
ln -s ../kinematics_learning/jacob/rgym
ln -s ../kinematics_learning/jacob/jacob
ln -s ../kinematics_learning/jacob/policy
pip install -e .
```
You might want to create a copy of the venv and install Zoo only in this copy to keep things clean.

