import json, pickle
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path as path
from matplotlib import colors as mcolors
import random

def get_label(n_dims, n_joints, lr, n_hidden, size_hidden, model):
    label = str(lr)+"_"+str(n_hidden)+"_"+str(size_hidden) 
    return label

def draw(loss_files, kwargs_files, group_by):
    fig = plt.figure()
    ax = fig.add_subplot()
    _c = dict(mcolors.XKCD_COLORS, **mcolors.XKCD_COLORS)
    colors = [c for c in _c ]
    random.shuffle(colors)
    i = 0
    best = dict() 

    for lf, kw in zip(loss_files, kwargs_files):
        with open(lf, 'rb') as f:
            losses, durations = pickle.load(f)

        with open(kw, 'r') as f:
            kwargs = json.load(f)
        label=get_label(**kwargs)
        n_epochs = losses.shape[1] 
        epochs = np.linspace(1, n_epochs, n_epochs)
        plt.fill_between(epochs, np.min(losses, axis=0), np.max(losses, axis=0), alpha=0.3, color=colors[i])
        
        mean_losses = np.mean(losses, axis=0)
        plt.plot(epochs, mean_losses, label=label, color=colors[i])
        i += 1

        ml = mean_losses[-10:].mean()
        if len(best) < 5:
            best[label] = ml 
        else:
            worst = max(best, key=best.get)
            if ml < best[worst]:
                best.pop(worst)
                best[label] = ml 
    best = dict(sorted(best.items(), key=lambda item: item[1]))

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    (path.cwd()/'results/tunning_fkine/eval').mkdir(parents=True,exist_ok=True)
    plt.text(0.0, -0.2, str(list(best.keys())), fontsize=12, transform=ax.transAxes)
    fig.savefig('results/tunning_fkine/eval/%s_%dd%dj.pdf'%(kwargs['model'], kwargs['n_dims'], kwargs['n_joints']), bbox_inches='tight')
    plt.close()
    return

if __name__ == "__main__":
    tunning_dir = path.cwd()/'results/tunning_fkine'

    models = ['FKineLinked', 'FKineMono']
    for n_dims in [2, 3]:
        for n_joints in [2, 3, 4, 5, 6, 7]:
            for model in models:
                loss_files = sorted(list(tunning_dir.glob('*%s_%dd%dj*.pickle'%(model, n_dims, n_joints))))
                kwargs_files = sorted(list(tunning_dir.glob('*%s_%dd%dj*_kwargs.json'%(model, n_dims, n_joints))))
                draw(loss_files, kwargs_files, '')
