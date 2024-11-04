import torch
from torch import nn
from typing import Optional, Dict, Union
import os
from avalanche.benchmarks import SplitCIFAR100
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.evaluation.metrics import (accuracy_metrics,
                                          forgetting_metrics,
                                          WandBStreamConfusionMatrix)
#from evaluation.metrics.flops import flops_metrics
from avalanche.training.plugins import EvaluationPlugin

from torchvision import transforms


from model_v2.conv_layers import ConvLayers
#from model_v2.classifier_F import Classifier

from model_v3.small_mobnet import ModelWrapper, build_classifier

from training.siesta_v4 import SIESTA

import argparse


# Flags for quechua
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(16)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HTTPS_PROXY"] = "http://icdvm14.ewi.tudelft.nl:3128"
#print(f"Number of GPUs visible: {torch.cuda.device_count()}")
#print(f"Current GPU: {torch.cuda.current_device()}")
#print(f"GPU name: {torch.cuda.get_device_name(0)}")


def load_model(
    path: str,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True
) -> nn.Module:
    """
    Load a PyTorch model from a .pth file

    Args:
        model: The model architecture to load weights into
        path: Path to the .pth file
        device: Device to load the model to ('cpu', 'cuda', or specific GPU)
        strict: Whether to strictly enforce that the keys in state_dict match

    Returns:
        Model with loaded weights
    """
    model = ConvLayers()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the state dict

    try:
        state_dict = torch.load(path, map_location=device)
        # Handle different save formats
        if isinstance(state_dict, Dict):
            if 'state_dict' in state_dict:  # Checkpoint contains more than just weights
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

        new_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, dict):  # If nested
                for sub_key, sub_value in value.items():
                    new_key = f"{key}.{sub_key}"
                    new_state_dict[new_key] = sub_value
            else:  # If not nested
                new_state_dict[key] = value

        # Load weights into model
        model.load_state_dict(new_state_dict, strict=strict)
        return model

    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def main(args):
    """
    try:
        classifier_G = load_model(
            path='checkpoints/conv_layers_cuda.pth',
            device='cuda',
            strict=True
        )
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Failed to load model: {e}")

    classifier_F = Classifier(init_features=2000, embed_size=args.embed_size,
                              num_classes=100, distance_measure="CosineLinear")

    """
    extract_from = f'model.features.{args.latent_layer - 1}'
    classifier_ckpt = 'checkpoints/swav_100c_2000e_mobilenet_modified_gelu_updated.pth'
    classifier_F = build_classifier('MobNet_ClassifierF', classifier_ckpt=classifier_ckpt,
                                    latent_layer=args.latent_layer, num_classes=100)
    core = build_classifier('MobNet_ClassifierG', classifier_ckpt=classifier_ckpt,
                            latent_layer=args.latent_layer, num_classes=100)

    classifier_G = ModelWrapper(core, output_layer_names=[
                                extract_from], return_single=True)
    # Define transforms for the CIFAR-100 datase
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 for MobileNetV3
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(
            0.2470, 0.2435, 0.2616))  # Normalization for CIFAR-10
    ])

    benchmark = SplitCIFAR100(
        n_experiences=10, seed=42, return_task_id=False, train_transform=transform,  eval_transform=transform)

    wandb_logger = WandBLogger(
        project_name=args.project_name,
        run_name=args.run_name,
        log_artifacts=False,
        config=vars(args)
    )

    eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True, epoch_running=True),
                                   forgetting_metrics(experience=True),
                                   # flops_metrics(profiler_step=(
                                   #   0, args.sleep_freq - 1)),
                                   loggers=[InteractiveLogger(), wandb_logger])

    strategy = SIESTA(
        classifier_G=classifier_G,
        classifier_F=classifier_F,
        pretrained=False,
        num_classes=100,
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
