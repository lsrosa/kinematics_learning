## Scripts

- tune_fkine.py - trains fkine with different hyperparams using ray tune
- learn_link_mono.py - given fkine hyperparams, train models using the Linked and Monolithic architectures 
- plot_loss_error_dist.py - plot loss and errors figures from the paper
- plot_x_xdot.py - plot x and xdot figures from the paper
- learn_marrtino.py - learn fkine for Marrtino model
- control_marrtino.py - uses the learned Marrtino model for implementing the Cartesian Impedance Controller

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
