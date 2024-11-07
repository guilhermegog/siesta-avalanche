import subprocess
import itertools

# Define hyperparameter grid
sleep_lr = [0.001]
sleep_freq = [2]
mem_size = [1000]
embed_size = [400]  # 1024 for small mobile net, 1280 for large
sleep_mb = [128]
sleep_iter = [1000]
latent_layer = [1,1,1,1,1,1,1,1,1,1]
split_net = False
project_name = "SplitMIST_SIESTA"


# Iterate over all combinations of hyperparameters
i = 0
for layer, lr, iter, freq, mem, embed, mb in itertools.product(latent_layer, sleep_lr, sleep_iter, sleep_freq, mem_size, embed_size, sleep_mb):
    run_name = 'smnist_embed_400'
    cmd = [
        "python", "siesta_smnist_exp.py",  # path to your main script
        "--project_name", project_name,
        "--run_name", run_name,
        '--split_net', str(split_net),
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
