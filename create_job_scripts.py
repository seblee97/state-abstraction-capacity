import os 
import itertools

lrs = [0.0001, 0.0003, 0.0005]
batch_sizes = [32, 128, 512]
buffer_sizes = [1024, 2048, 4096]
kls = [0.01, 0.02, 0.03]
epochs = [3,5,7]
entropies = [0.01, 0.02, 0.03]

# Create a directory for job scripts if it doesn't exist
os.makedirs("job_scripts", exist_ok=True)
job_script_template = """#!/bin/bash\n
#SBATCH -p genx\n
#SBATCH --nodes 1\n
#SBATCH --cpus-per-task 4\n
#SBATCH --mem 4G\n
#SBATCH --time=0-05:00:00\n
#SBATCH --output=/mnt/home/slee1/ceph/unc_mattar_results/2025-10-08-16-41-40/dyna_0/0/output.txt\n
#SBATCH --error=/mnt/home/slee1/ceph/unc_mattar_results/2025-10-08-16-41-40/dyna_0/0/error.txt\n
source /mnt/home/slee1/venvs/sac/bin/activate\n
"""

for idx, (lr, batch_size, buffer_size, kl, epoch, entropy) in enumerate(
    itertools.product(lrs, batch_sizes, buffer_sizes, kls, epochs, entropies)
):
    # create subdirectory for each job script
    os.makedirs(f"job_scripts/job_{idx}", exist_ok=True)
    with open(f"job_scripts/job_{idx}/job_{idx}.sh", "w") as f:
        f.write(job_script_template)
        f.write(f"python /mnt/home/slee1/state-abstraction-capacity/sac/run.py -m ppo -lr {lr} -bs {batch_size} -buf {buffer_size} -kl {kl} -ep {epoch} -ent {entropy}\n")

    os.chmod(f"job_scripts/job_{idx}/job_{idx}.sh", 0o755)
    