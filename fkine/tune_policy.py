import sys
import os
import json
from pathlib import Path as path
from rl_zoo3.train_sac_fe import train_sac_fe
import torch

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

def tune(folder):
    sys.argv = ["python",
            "--algo",
            "sac",
            "--env",
            "ReacherPolicy",
            "--n-jobs",
            "2",
            "-n",
            "50000",
            "-optimize",
            "--n-trials",
            "100",
            "--n-eval-envs",
            "32",
            "--pruner",
            "median",
            "--log-folder",
            folder,
            "--device",
            str(device),
            ]

    train_sac_fe()
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

def tune_and_report(models_folder, fkine_models_folder, n_dims, n_joints):
    model_files = sorted(list(path(models_folder).glob('reacher%dd%dj*.xml'%(n_dims, n_joints))))
    default_model_file = 'rgym/envs/assets/default.xml'
    default_fkine_model_file = fkine_models_folder+'/default_model.pt'
    default_fkine_kwargs_file = fkine_models_folder+'/default_kwargs.json'
    
    log_folder = 'results/policy/tunning/'

    # copy the model file to the default
    for model_file in model_files:
        os.system("cp %s %s"%(model_file.as_posix(), default_model_file))
        model_name = model_file.parts[-1].replace('.xml', '')
        # search for fkine model and kwargs for model_name
        fkine_model_files = sorted(list(path(fkine_models_folder).glob('*%s*.pt'%model_name.replace('reacher',''))))
        fkine_kwargs_files = sorted(list(path(fkine_models_folder).glob('*%s*.json'%model_name.replace('reacher',''))))
        for fkine_model_file, fkine_kwargs_file in zip(fkine_model_files, fkine_kwargs_files):
            print(fkine_model_file)
            print(fkine_kwargs_file)
            os.system("cp %s %s"%(fkine_model_file, default_fkine_model_file))
            os.system("cp %s %s"%(fkine_kwargs_file, default_fkine_kwargs_file))
            with open(default_fkine_kwargs_file, 'r') as f:
                fkine_kwargs = json.load(f)
                print('fkine model: ', fkine_kwargs['model'])
            tune(log_folder+model_name+fkine_kwargs['model'])
            report(log_folder+model_name+fkine_kwargs['model'])
    return

if __name__ == "__main__":
    models_foder = "rgym/envs/assets"
    fkine_models_folder = "results/fkine_models"
    for n_dims in [3]:#[2, 3]:
        for n_joints in [7]:#[2, 3, 4, 5, 6, 7]:    
            tune_and_report(models_foder, fkine_models_folder, n_dims, n_joints)
    
