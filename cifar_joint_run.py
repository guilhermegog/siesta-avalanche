import subprocess
import itertools

# Define hyperparameter grid
learning_rate = [0.1, 0.01, 0.001]
arch_list = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"]
optim_list = ["SGD", "Adam", "RMSprop"]

# Iterate over all combinations of hyperparameters
i = 0
for lr, arch, optim in itertools.product(learning_rate, arch_list, optim_list):
    cmd = [
        "python", "cifar_joint.py",  # path to your main script
        "--lr", str(lr),
        "--arch", arch,
        "--optim", optim
    ]
    subprocess.run(cmd)
    i = i+1
