import torch
from torch import nn
from tqdm import tqdm
import sys
import os
from torchvision.transforms import v2

from utils import (set_seed, get_imagenet_subset,
                   adapt_mbnet, load_partial_model, save_partial_model
                   )

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(f"Number of GPUs visible: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")


def accuracy(output, target, topk=(1, 5)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train(model, train_loader, optimizer, criterion, device, epoch):

    model.train()
    running_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))
    ])
    cutmix = v2.CutMix(num_classes=100, alpha=1.0)
    mixup = v2.MixUp(num_classes=100, alpha=0.1)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = cutmix_or_mixup(inputs, labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (labels.shape != 2):
            labels, indices = torch.max(labels, dim=1)

        top1, top5 = accuracy(outputs, indices)

        running_loss += loss.item()
        total_samples += labels.size(0)
        total_top1 += top1
        total_top5 += top5
        if i % 10 == 0:
            tqdm.write(f"Epoch [{epoch+1}] Train - Batch [{i}/{len(train_loader)}] Loss: {
                       running_loss/total_samples:.4f} Acc-T1: {total_top1/total_samples:.4f} Acc-T5: {total_top5/total_samples:.4f}")
            sys.stdout.flush()

    train_loss = running_loss / len(train_loader)
    train_acc_top1 = total_top1 / total_samples
    train_acc_top5 = total_top5/total_samples
    return train_loss, train_acc_top1, train_acc_top5


def evaluate(model, test_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_top1 = 0.0
    total_top5 = 0.0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))

            running_loss += loss.item()
            total += labels.size(0)
            total_top1 += top1
            total_top5 += top5
            if i % 100 == 0:
                tqdm.write(f"Epoch [{epoch+1}] Train - Batch [{i}/{len(test_loader)}] Loss: {
                    running_loss/total:.4f} Acc-T1: {total_top1/total:.4f} Acc-T5: {total_top5/total:.4f}")
                sys.stdout.flush()

    test_loss = running_loss / len(test_loader)
    test_acc_top1 = total_top1 / total
    test_acc_top5 = total_top5 / total
    return test_loss, test_acc_top1, test_acc_top5


def create_param_groups(model):
    # Example: Splitting parameters into two groups: base layers (features) and classifier
    param_groups = []
    current_lr = 0.2
    for i, (name, param) in enumerate(model.named_parameters()):
        if (param.requires_grad == True):
            param_groups.append({
                'params': [param],
                'lr': current_lr
            })
            current_lr *= 0.99

    return param_groups


def main():
    set_seed(42)  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-100 subset
    train_loader, test_loader, selected_classes = get_imagenet_subset(
        num_classes=100, batch_size=256)

    # Create and load the model
    model = adapt_mbnet(num_classes=100).to(device)
    model = load_partial_model(
        model, 'checkpoints/best_cosine_softmax_loss_SWAV_sgd_layerlr02_step_MIXUP_CUTMIX_50e_100c.pth', num_frozen_layers=8)

    param_groups = create_param_groups(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        param_groups, lr=0.2, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1)
    num_epochs = 100
    best_acc = 0.0

    # MixUp and CutMix

    for epoch in range(num_epochs):
        tqdm.write(f"\nEpoch [{epoch+1}/{num_epochs}]")
        sys.stdout.flush()

        train_loss, train_acc_top1, train_acc_top5 = train(
            model, train_loader, optimizer, criterion, device, epoch)
        test_loss, test_acc_top1, test_acc_top5 = evaluate(
            model, test_loader, criterion, device, epoch)

        scheduler.step()
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        tqdm.write(f"Train Loss: {train_loss:.4f}, Train Acc: {
                   train_acc_top1:.4f}")
        tqdm.write(f"Test Loss: {test_loss:.4f}, Test Acc: {
                   test_acc_top1:.4f}")
        tqdm.write(f"Train Loss: {train_loss:.4f}, Train Acc Top 5: {
                   train_acc_top5:.4f}")
        tqdm.write(f"Test Loss: {test_loss:.4f}, Test Acc Top 5: {
                   test_acc_top5:.4f}")
        sys.stdout.flush()

        # Save the state dictionary if we have a new best accuracy
        if test_acc_top5 > best_acc:
            best_acc = test_acc_top5
            save_partial_model(
                model, 'checkpoints/g_net.pth', start_layer=8)
            tqdm.write(f"\nSaved state dictionary from 9th layer up. Best accuracy: {
                       best_acc:.4f}")
            sys.stdout.flush()

    print("Selected ImageNet1k classes:", selected_classes)


if __name__ == "__main__":
    main()
