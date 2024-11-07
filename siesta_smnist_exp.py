import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from typing import Optional, Dict, Union
import os
import copy
from avalanche.benchmarks import SplitMNIST
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.evaluation.metrics import (accuracy_metrics,
                                          forgetting_metrics,
                                          WandBStreamConfusionMatrix)
#from evaluation.metrics.flops import flops_metrics
from avalanche.training.plugins import EvaluationPlugin


from model_v2.classifier_F import CosineLinear
#from model_v2.classifier_F import Classifier

#from model_v3.small_mobnet import ModelWrapper, build_classifier
from model_v3.mobnetv3_small import ModelWrapper, build_classifier


from training.siesta_v4 import SIESTA

import argparse


# Flags for quechua
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(16)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HTTPS_PROXY"] = "http://icdvm14.ewi.tudelft.nl:3128"
#print(f"Number of GPUs visible: {torch.cuda.device_count()}")
#print(f"Current GPU: {torch.cuda.current_device()}")
#print(f"GPU name: {torch.cuda.get_device_name(0)}")


#torch.manual_seed()


class MNISTNetwork(nn.Module):
    def __init__(self):
        super(MNISTNetwork, self).__init__()
        # Input layer for MNIST images (784 input features)
      # Two hidden layers with 400 ReLU neurons each
        # Output layer with 10 classes for MNIST

    def forward(self, x):
        # Flatten the input image (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(-1, 784)
        return x


class FClass(nn.Module):
    def __init__(self, embed_size):
        super(FClass, self).__init__()
        self.input_layer = nn.Linear(784, 400)
        self.hidden1 = nn.Linear(400, 400)
        self.layer = nn.Linear(400, embed_size)
        self.output = CosineLinear(embed_size, 10)

    def forward(self, x, feat=False):
        x = F.relu(self.input_layer(x))
        x = self.hidden1(x)
        x = F.relu(x)
        feature = self.layer(x)
        x = F.relu(feature)
        out = self.output(x)
        if feat:
            return feature, out

        return out


def load_model() -> nn.Module:
    classifier_G = MNISTNetwork()
    classifier_F = FClass(400)
    return classifier_G, classifier_F


def main(args):
    classifier_G, classifier_F = load_model()

    benchmark = SplitMNIST(
        n_experiences=5, return_task_id=False)

    wandb_logger = WandBLogger(
        project_name=args.project_name,
        run_name=args.run_name,
        log_artifacts=False,
        config=vars(args)
    )

    eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True),
                                   forgetting_metrics(experience=True),
                                   # flops_metrics(profiler_step=(
                                   #   0, args.sleep_freq - 1)),
                                   loggers=[InteractiveLogger(), wandb_logger])

    strategy = SIESTA(
        classifier_G=classifier_G,
        classifier_F=classifier_F,
        pretrained=False,
        num_classes=10,
        embed_size=args.embed_size,
        lr=args.sleep_lr,
        sleep_n_iter=args.sleep_iter,
        sleep_frequency=args.sleep_freq,
        sleep_mb_size=args.sleep_mb,
        eval_mb_size=256,
        memory_size=args.mem_size,
        device="cuda",
        evaluator=eval_plugin
    )

    # Train the model
    results = []
    eval_experiences = []
    for t_experience, e_experience in zip(benchmark.train_stream, benchmark.test_stream):
        print(f"Start of experience: {t_experience.current_experience}")

        strategy.train(experiences=t_experience)

        print(f"End of experience: {t_experience.current_experience}")
        eval_experiences.append(e_experience)
        for eval_exp in eval_experiences:
            results.append(strategy.eval(eval_exp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, help="WandB project name")
    parser.add_argument('--run_name', type=str, help='Current run name')
    parser.add_argument('--split_net', type=bool,
                        help='Sleep train on entire or split net')
    parser.add_argument('--latent_layer', type=int,
                        help='Layer from which LRs are extracted', default=8)
    parser.add_argument('--sleep_lr', type=float, help="LR during sleep")
    parser.add_argument('--sleep_iter', type=int,
                        help='Iterations during sleep phase')
    parser.add_argument('--sleep_freq', type=int,
                        help="Number of exps between sleeps")
    parser.add_argument('--mem_size', type=int,
                        help='Number of latent activations in buffer')
    parser.add_argument('--embed_size', type=int, help="Prototype size")
    parser.add_argument('--sleep_mb', type=int, help='MB_size during sleep')
    parser.add_argument('--run_nr', type=int, default=0,
                        help="A top-level flag to control where in the hyperparameter space the run is")
    args = parser.parse_args()
    main(args)
