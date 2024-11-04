import subprocess
import itertools

# Define hyperparameter grid
sleep_lr = [0.0000001]
sleep_freq = [5]
mem_size = [20000]
embed_size = [1280]
sleep_mb = [256]
sleep_iter = [2500]
latent_layer = [8]
project_name = "SplitCIFAR100_mbnetv3"


# Iterate over all combinations of hyperparameters
i = 0
for layer, lr, iter, freq, mem, embed, mb in itertools.product(latent_layer, sleep_lr, sleep_iter, sleep_freq, mem_size, embed_size, sleep_mb):
    run_name = 'scifar_run_iter_fclass'
    cmd = [
        "python", "siesta_cifar_exp.py",  # path to your main script
        "--project_name", project_name,
        "--run_name", run_name,
        "--latent_layer", str(layer),
        "--sleep_iter", str(iter),
        "--sleep_lr", str(lr),
        "--sleep_freq", str(freq),
        "--mem_size", str(mem),
        "--embed_size", str(embed),
        "--sleep_mb", str(mb),
        "--run_nr", str(i),

    ]
    subprocess.run(cmd)
    i = i+1
