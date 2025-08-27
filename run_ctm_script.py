import os
import re
import numpy as np
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json

@click.command()
@click.option('-c', '--task_config_name', required=True, type=str)
@click.option('-p', '--path', required=True)
@click.option('-d', '--device', required=True, type=str)
@click.option('-hdd', '--hdd_dir', required=False, type=str)
def main(task_config_name, path, device='cuda:0', hdd_dir="outputs"):
    '''
    From given path e.g. "outputs/hammer_lowdim_human-v3_reproduction/train_by_seed"
    find all the best checkpoints and run eval.py for each checkpoint
    '''
    assert task_config_name in path, f"task_config_name {task_config_name} not in path {path}"
    
    # Regex to extract score from filename like 'epoch=1000-test_mean_score=0.560.ckpt'
    score_pattern = re.compile(r"test_mean_score=([\d\.eE+-]+)\.ckpt")

    for seed_number in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        # Find all directories starting with f"seed_{seed_number}" and get the last one
        seed_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith(f"seed_{seed_number}")]
        if not seed_dirs:
            print(f"\033[91mNo directories found starting with seed_{seed_number}\033[0m")
            continue
        entry = sorted(seed_dirs)[-1]  # Get the last directory alphabetically
        
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            checkpoints_dir = os.path.join(full_path, "checkpoints")
            if not os.path.isdir(checkpoints_dir):
                continue
            checkpoint_list = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt') and 'test_mean_score=' in f])
            
            
            max_score = None
            max_file = None
            for fname in checkpoint_list:
                match = score_pattern.search(fname)
                if match:
                    try:
                        score = float(match.group(1))
                        if max_score is None or score > max_score:
                            max_score = score
                            max_file = fname
                    except ValueError:
                        continue
            print(f"\033[92m{entry}: {max_file} (score={max_score})\033[0m")
            max_file = max_file.replace("=", "=")
            
            run_dir=f"{hdd_dir}/${{task.name}}_${{task.dataset_type}}_reproduction/train_by_seed_ctm/seed_${{training.seed}}_${{now:%Y.%m.%d-%H.%M.%S}}_${{name}}_${{task_name}}_cnn_${{horizon}}"

            command = f"python train.py --config-dir=configs/ --config-name=ctmp_{task_config_name}_lowdim.yaml hydra.run.dir={run_dir} logging.group='${{task.name}}_${{task.dataset_type}}_ctm' logging.name='seed_${{training.seed}}_ctm_${{now:%Y.%m.%d-%H.%M.%S}}_${{name}}_${{task_name}}_${{horizon}}' horizon=32 task.dataset.horizon=32 task.dataset.pad_after=31 training.seed=$seed training.device={device} obs_as_global_cond=True training.num_epochs=200 policy.horizon=32 training.debug=false policy.teacher_path={max_file} policy.edm={max_file}"
            print(command)
            os.system(command)

if __name__ == '__main__':
    main()