import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

from model_v3.mobnetv3_small import CosineLinear
import numpy as np
import os

from training.create_mobnet import create_classifiers

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import wandb
import argparse

# Flags for quechua
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.environ["HTTPS_PROXY"] = "http://icdvm14.ewi.tudelft.nl:3128"


def get_layerwise_params(classifier, lr):
    trainable_params = []
    layer_names = []
    lr_mult = 0.9  # 0.99
    for idx, (name, param) in enumerate(classifier.named_parameters()):
        layer_names.append(name)
    # reverse layers
    layer_names.reverse()
    # store params & learning rates
    for idx, name in enumerate(layer_names):
        # append layer parameters
        trainable_params += [
            {
                "params": [p for n, p in classifier.named_parameters() if n == name and p.requires_grad
                           ],
                "lr": lr,
            }
        ]
        # update learning rate
        lr *= lr_mult
    return trainable_params


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),  # Random crop with padding
    transforms.RandomHorizontalFlip(),     # Random horizontal flip
    # Convert to tensor
    # Normalize with CIFAR-10 mean and std
    transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                         std=(0.2673, 0.2564, 0.2762))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                         std=(0.2673, 0.2564, 0.2762))
])

# Load CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR100(
    root='/space/gguedes/datasets/cifar100/data', train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR100(
    root='/space/gguedes/datasets/cifar100/data', train=False, download=True, transform=transform_test
)

# DataLoaders
trainloader = DataLoader(trainset, batch_size=128,
                         shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)


def train(classifier_G, classifier_F, epoch, optimizer, criterion):
    classifier_G.eval()
    classifier_F.train()

    classifier_G.cuda()
    classifier_F.cuda()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = classifier_G(inputs)
        outputs = classifier_F(outputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update training metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}: [{batch_idx}/{len(trainloader)}] '
                  f'Loss: {train_loss/(batch_idx+1):.3f} '
                  f'Acc: {100.*correct/total:.3f}%')

    return 100.*correct/total, train_loss / len(trainloader)

# Validation Loop


def validate(classifier_G, classifier_F, epoch, criterion):
    classifier_G.eval()
    classifier_F.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = classifier_G(inputs)
            outputs = classifier_F(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f'Validation Epoch {epoch}: '
          f'Loss: {test_loss/len(testloader):.3f} '
          f'Accuracy: {accuracy:.3f}%')

    return accuracy, test_loss/len(testloader)


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def main(args):
    user = 'gguedes'
    project = 'CIFAR100_joint'
    latent_layer = 8
    disp_name = f'{args.arch}_{args.optim}_{args.lr}'
    early_stopping = EarlyStopping(patience=10, delta=0.01)
    if args.arch == 'mobilenet_v3_small' or args.arch == 'mobilenet_v3_large':
        from model_v4.mobnet import MobNet_ClassifierG, MobNet_ClassifierF
        if args.arch == 'mobilenet_v3_small':
            embed_size = 1024
            latent_layer = 5
    else:
        from model_v4.effnet import MobNet_ClassifierG, MobNet_ClassifierF

    classifier_G = MobNet_ClassifierG(latent_layer, args.arch)
    classifier_F = MobNet_ClassifierF(
        latent_layer=latent_layer, num_classes=100, arch=args.arch)

    config = {"lr": args.lr,
              "arch": args.arch}

    wandb.init(entity=user, project=project, name=disp_name, config=config)

    best_accuracy = 0

    params = get_layerwise_params(classifier_F, args.lr)
    if(args.optim == "SGD"):
        optimizer = optim.SGD(params,
                              lr=args.lr, weight_decay=0.0001, momentum=0.9)
    elif(args.optim == "Adam"):
        optimizer = optim.Adam(params, lr=args.lr)
    else:
        optimizer = optim.RMSprop(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.95)

    # Create directory for saving models if it doesn't exist
    loss_array = []
    val_array = []
    # Training loop
    for epoch in range(0, 50):  # 100 epochs
        train_acc, train_loss = train(
            classifier_G, classifier_F, epoch, optimizer, criterion)
        val_accuracy, val_loss = validate(
            classifier_G, classifier_F, epoch, criterion)
        lr_scheduler.step()

        loss_array.append(train_loss)
        val_array.append(val_accuracy)
        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

        wandb.log({'train_acc': train_acc,
                   "train_loss": train_loss,
                   "val_acc": val_accuracy,
                   "val_loss": val_loss})
        early_stopping(val_loss, classifier_F)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # Save final model
    print(
        f'Training completed. Best validation accuracy: {best_accuracy:.2f}%')

    wandb.finish()


# Run the training
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--arch', type=str, default="mobilenet_v3_small")
    parser.add_argument('--optim', type=str, default='SGD')
    args = parser.parse_args()
    main(args)
