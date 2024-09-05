import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.siesta_net import SiestaNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define transforms for the CIFAR-100 dataset
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ]
)

# Load the CIFAR-100 training dataset
train_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

single_input, single_target = train_dataset[50]

model = SiestaNet(num_classes=100).to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for batch_idx, (inputs, targets) in enumerate(train_loader):
    if single_input in inputs:
        pass
    optimizer.zero_grad()
    inputs, targets = inputs.to(device), targets.to(device)
    outputs,z, lr = model(inputs)
    # Apply log to outputs because NLLLoss expects log probabilities
    log_outputs = torch.log(outputs)
    loss = criterion(log_outputs, targets)
    optimizer.step()

    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}: Loss {loss.item()}")

# Test the online learning function
print(single_target)
output, z, lr = model(single_input.unsqueeze(0).to(device))
model.f_classifier.online_update(z, single_target)
