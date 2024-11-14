import subprocess
import itertools

# Define hyperparameter grid
learning_rate = [0.1, 0.01, 0.005, 0.001]

# Iterate over all combinations of hyperparameters
i = 0
for lr in learning_rate:
    cmd = [
        "python", "cifar_joint.py",  # path to your main script
        "--lr", str(lr),
    ]
    subprocess.run(cmd)
    i = i+1
