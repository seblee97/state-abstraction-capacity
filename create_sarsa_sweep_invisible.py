import os
import itertools
import subprocess
import time
import numpy as np

# Sweep parameters
lrs = [0.00003, 0.0001, 0.0003]
lambdas = [0.8, 0.9, 0.95]

# Fixed
timeout = 2000
eps_decay = 0.99999
opt = 1.0
num_ep = 20000
eps = 1.0
gamma = 0.99
test_freq = 100
viz_freq = 2000
save_freq = 2000
stats_save_freq = 500

PROJECT_DIR = "/mnt/home/slee1/projects/state-abstraction-capacity"
VENV = "/mnt/home/slee1/venvs/sac/bin/activate"
BASE_RESULTS = "/mnt/home/slee1/ceph/sarsa_mdp"

os.makedirs(BASE_RESULTS, exist_ok=True)
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
sweep_dir = os.path.join(BASE_RESULTS, timestamp)
os.makedirs(sweep_dir, exist_ok=True)

job_script_template = """#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 8G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
"""

sweep_config = {}
idx = 0

for lr, lam in itertools.product(lrs, lambdas):
    config = {
        "map_yaml": "shaped_meister_trimmed.yaml",
        "lr": lr,
        "lambda_": lam,
        "timeout": timeout,
        "eps_decay": eps_decay,
        "optimistic_init": opt,
        "num_ep": num_ep,
        "invisible_rewards": True,
    }
    sweep_config[idx] = config

    job_dir = os.path.join(sweep_dir, f"job_{idx}")
    os.makedirs(job_dir, exist_ok=True)

    job_path = os.path.join(job_dir, f"job_{idx}.sh")
    with open(job_path, "w") as f:
        f.write(job_script_template)
        f.write(f"#SBATCH --job-name=invis_lr{lr}_lam{lam}\n")
        f.write(f"#SBATCH --output={job_dir}/output.txt\n")
        f.write(f"#SBATCH --error={job_dir}/error.txt\n")
        f.write(f"source {VENV}\n")
        f.write(f"cd {PROJECT_DIR}\n")
        f.write(
            f"python -m sac.run"
            f" -m deep_sarsa_lambda"
            f" -conv"
            f" -invis"
            f" -abs_results {job_dir}"
            f" -map_yaml shaped_meister_trimmed.yaml"
            f" -test_map_yaml test_meister_trimmed.yaml"
            f" -lr {lr}"
            f" -lam {lam}"
            f" -timeout {timeout}"
            f" -eps {eps}"
            f" -eps_decay {eps_decay}"
            f" -gamma {gamma}"
            f" -opt {opt}"
            f" -num_ep {num_ep}"
            f" -test {test_freq}"
            f" -viz {viz_freq}"
            f" -save {save_freq}"
            f" -save_stats {stats_save_freq}"
            f" -es 5000\n"
        )

    os.chmod(job_path, 0o755)
    subprocess.call(f"sbatch {job_path}", shell=True)
    idx += 1

np.save(os.path.join(sweep_dir, "sweep_config.npy"), sweep_config)
print(f"Submitted {idx} jobs. Config saved to {sweep_dir}/sweep_config.npy")
