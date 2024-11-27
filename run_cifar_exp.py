import subprocess
import itertools

# Define hyperparameter grid
sleep_lr = [0.2, 0.002]
sleep_freq = [2]
mem_size = [20000]
embed_size = [1024]  # 1024 for small mobile net, 1280 for large
sleep_mb = [16]
sleep_iter = [4000]
latent_layer = [3]
arch_list = ['mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0']
project_name = "SplitCIFAR100_arch_sweep"
finetune = ["False", "True"]

# Iterate over all combinations of hyperparameters
i = 0
for layer, lr, iter, freq, mem, embed, mb, arch, ft in itertools.product(latent_layer, sleep_lr, sleep_iter, sleep_freq, mem_size, embed_size, sleep_mb, arch_list, finetune):
    run_name = 'scifar_' + arch
    cmd = [
        "python", "siesta_cifar_exp.py",  # path to your main script
        "--project_name", project_name,
        "--run_name", run_name,
        '--arch', arch,
        '--finetune', ft,
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
