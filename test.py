from benchmarks.siesta_benchmark import make_cl_benchmark
from avalanche.benchmarks.classic import SplitImageNet
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from evaluation.metrics.my_accuracy import accuracy_metrics_top5
from training.siesta_v3 import SIESTA
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch
import os

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(16)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(f"Number of GPUs visible: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")


def main():
    label_dir = '/space2/gguedes/Documents/siesta-avalanche/benchmarks/'
    arr = np.load(label_dir + '/imagenet_files/combined_array.npy')
    arr = arr.astype(int)
    print(len(arr))
    exp_dict = {}
    n_exp = 19
    start = 0
    for i in range(n_exp):
        if i == 0:
            end = 100
        else:
            end = 50
        end_ix = start+end

        exp_dict[i] = arr[start:end_ix]
        start = end_ix
    n_exp = 19
    start = 0
    for i in range(n_exp):
        if i == 0:
            end = 100
        else:
            end = 5

        exp_dict[i] = arr[start:start+end]
        exp_dict = {}
        exp_dict[0] = 100
        start = end_ix
    result = arr.tolist()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #    transforms.RandomResizedCrop(224),
        #    transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    wandb_logger = WandBLogger(
        project_name="siesta_avalanche",
        run_name="base_run_imagenet1k",
        log_artifacts = False,
    )

    benchmark = SplitImageNet(dataset_root="/space/gguedes/datasets/imagenet1k",
                              n_experiences=19,
                              per_exp_classes=exp_dict,
                              fixed_class_order=result,
                              return_task_id=False,
                              seed=42,
                              train_transform=transform,
                              eval_transform=transform
                              )
    eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True, epoch_running=True),
                                   accuracy_metrics_top5(epoch = True, experience=True, epoch_running=True),
                                   loggers=[InteractiveLogger(), wandb_logger])

    strategy = SIESTA(num_classes=1000, criterion=torch.nn.CrossEntropyLoss(), lr=1.6, tau=1, seed=None, sleep_frequency=2, sleep_mb_size=512, eval_mb_size=1024, memory_size=959665,
                      device="cuda", evaluator=eval_plugin, eval_every=0)
    print('Starting experiment...')
    results = []
    eval_experiences = []
    for t_experience, e_experience in zip(benchmark.train_stream, benchmark.test_stream):
        res = strategy.train(t_experience)
        print('Training Completed')
        print('Computing accuracy on the whole test set')
        eval_experiences.append(e_experience)
        for eval_exp in eval_experiences:
            results.append(strategy.eval(eval_exp))


if __name__ == "__main__":
    main()
