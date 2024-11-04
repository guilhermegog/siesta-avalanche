import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
from small_mobnet import mobilenet_v3_small

# Flags for quechua
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(16)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HTTPS_PROXY"] = "http://icdvm14.ewi.tudelft.nl:3128"

model = mobilenet_v3_small()

# Set random seed for reproducibility
torch.manual_seed(42)

# Data Augmentation and Normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Random crop with padding
    transforms.RandomHorizontalFlip(),     # Random horizontal flip
    transforms.ToTensor(),                 # Convert to tensor
    # Normalize with CIFAR-10 mean and std
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(
    root='/space/gguedes/datasets/cifar10/data', train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root='/space/gguedes/datasets/cifar10/data', train=False, download=True, transform=transform_test
)

# DataLoaders
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


class AdaptedMobileNetV3Small(nn.Module):
    def __init__(self, num_classes=10):
        super(AdaptedMobileNetV3Small, self).__init__()

        # Load pre-trained MobileNetV3 Small
        mobilenetv3_small = mobilenet_v3_small()

        # Modify the first convolution layer to accept 32x32 CIFAR-10 images
        # Original first layer is designed for ImageNet (224x224), so we'll adjust
        original_first_conv = mobilenetv3_small.features[0]

        # Create a new first convolution layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=original_first_conv.out_channels,
                kernel_size=3,  # Smaller kernel for smaller images
                stride=1,        # Reduced stride
                padding=1,       # Same padding to maintain spatial dimensions
                bias=False
            ),
            nn.BatchNorm2d(original_first_conv.out_channels),
            # Keep the original hardswish activation
            mobilenetv3_small.features[0][2]
        )

        # Replace the first layer in the original model
        self.features = nn.Sequential(
            self.first_conv,
            *list(mobilenetv3_small.features.children())[1:]
        )

        # Modify the classifier to match CIFAR-10 classes
        self.classifier = mobilenetv3_small.classifier

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x


model = AdaptedMobileNetV3Small().cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01)

lr_scheduler = StepLR(optimizer, step_size=25, gamma=0.8)
# Training Loop


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
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

    return train_loss / len(trainloader)

# Validation Loop


def validate(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f'Validation Epoch {epoch}: '
          f'Loss: {test_loss/len(testloader):.3f} '
          f'Accuracy: {accuracy:.3f}%')

    return accuracy

# Main Training Script


def main():
    best_accuracy = 0

    # Create directory for saving models if it doesn't exist
    os.makedirs('./saved_models', exist_ok=True)

    # Training loop
    for epoch in range(1, 200):  # 100 epochs
        train_loss = train(epoch)
        val_accuracy = validate(epoch)
        lr_scheduler.step()
        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(),
                       './saved_models/best_cifar10_model.pth')

    # Save final model
    torch.save(model.state_dict(), './saved_models/final_cifar10_model.pth')
    print(
        f'Training completed. Best validation accuracy: {best_accuracy:.2f}%')


# Run the training
if __name__ == '__main__':
    main()
