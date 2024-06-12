import sys
import os
from pathlib import Path as path
from rl_zoo3.train import train

def tune(folder):
    sys.argv = ["python",
            "--algo",
            "sac",
            "--env",
            "ReacherPolicy",
            "--n-jobs",
            "16",
            "-n",
            "1000",
            "-optimize",
            "--n-trials",
            "100",
            "--pruner",
            "median",
            "--log-folder",
            folder
            ]

    train()
    return

def report(folder):
    study_pkl_path = list(path(folder+"/sac/").glob("*ReacherPolicy*.pkl"))[0].as_posix()
    command = ["python "+
              "../../rl-baselines3-zoo/scripts/parse_study.py "+
              "-i "+
              study_pkl_path+
              " --folder "+
              folder +
              " --print-n-best-trials "+
              "10 "+
              "--save-n-best-hyperparameters "+
              "1"][0]
    print(command)
    os.system(command)
    return

def tune_and_report(models_folder, n_dims, n_joints):
    model_files = sorted(list(path(models_folder).glob('reacher%dd%dj*.xml'%(n_dims, n_joints))))
    default_file = 'rgym/envs/assets/default.xml'
    
    log_folder = 'results/policy/tunning/'

    # copy the model file to the default
    for model_file in model_files:
        os.system("cp %s %s"%(model_file.as_posix(), default_file))
        results_folder = model_file.parts[-1].replace('.xml', '')
        tune(log_folder+results_folder)
        report(log_folder+results_folder)
    return

if __name__ == "__main__":
    models_foder = "rgym/envs/assets"
    for n_dims in [2]:#[2, 3]:
        for n_joints in [2]:#[2, 3, 4, 5, 6, 7]:    
            tune_and_report(models_foder, n_dims, n_joints)
    
