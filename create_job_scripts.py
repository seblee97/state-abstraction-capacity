import os 
import itertools
import subprocess
import time
import numpy as np

lrs = [0.0001, 0.0003, 0.0005]
batch_sizes = [32, 128, 512]
buffer_sizes = [1024, 2048, 4096]
kls = [0.01, 0.02, 0.03]
epochs = [3,5,7]
entropies = [0.01, 0.02, 0.03]

# create a directory for job scripts if it doesn't exist
os.makedirs("job_scripts", exist_ok=True)
# create timestamped subdirectory for this sweep
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(f"job_scripts/{timestamp}", exist_ok=True)

job_script_template = """#!/bin/bash
#SBATCH -p genx
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 4G
#SBATCH --time=0-05:00:00
"""

sweep_config = {}

for idx, (lr, batch_size, buffer_size, kl, epoch, entropy) in enumerate(
    itertools.product(lrs, batch_sizes, buffer_sizes, kls, epochs, entropies)
):
    sweep_config[idx] = {
        "lr": lr,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "kl": kl,
        "epoch": epoch,
        "entropy": entropy,
    }
    # create subdirectory for each job script
    job_dir = os.path.join(f"job_scripts/{timestamp}", f"job_{idx}")
    os.makedirs(job_dir, exist_ok=True)
    with open(f"{job_dir}/job_{idx}", "w") as f:
        f.write(job_script_template)
        f.write(f"#SBATCH --output={job_dir}/output.txt\n")
        f.write(f"#SBATCH --error={job_dir}/error.txt\n")
        f.write("source /mnt/home/slee1/venvs/sac/bin/activate\n")
        f.write(f"python /mnt/home/slee1/state-abstraction-capacity/sac/run.py -m ppo -rep pixel -conv -lr {lr} -bs {batch_size} -rbs {buffer_size} -kl {kl} -ue {epoch} -ec {entropy}\n")

    # make the script executable
    os.chmod(f"job_scripts/{timestamp}/job_{idx}/job_{idx}", 0o755)

    # run the command to submit the job
    subprocess.call(f"sbatch job_scripts/{timestamp}/job_{idx}/job_{idx}", shell=True)

np.save(f"job_scripts/{timestamp}/sweep_config.npy", sweep_config)