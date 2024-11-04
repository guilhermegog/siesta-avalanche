import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the network architecture


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(
            256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = torch.relu(x)

        x = self.bn2(self.conv2(x))
        x = torch.relu(x)

        x = self.bn3(self.conv3(x))
        x = torch.relu(x)

        x = self.bn4(self.conv4(x))
        x = torch.relu(x)

        x = self.bn5(self.conv5(x))
        x = torch.relu(x)

        x = x.view(-1, 512 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':

    # Define the transformations for the train and test sets
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the CIFAR-10 datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=12)
    # Instantiate the network, loss function, and optimizer
    device = torch.device('cuda:2' if torch.cuda.is_available()
                          else 'cpu')
    print(f'Using device: {device}')
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # Train the network
    for epoch in range(100):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(
                device, dtype=torch.float), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(
                    f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')
    conv_layers_state_dict = {
        'conv1': net.conv1.state_dict(),
        'bn1': net.bn1.state_dict(),
        'conv2': net.conv2.state_dict(),
        'bn2': net.bn2.state_dict(),
        'conv3': net.conv3.state_dict(),
        'bn3': net.bn3.state_dict(),
        'conv4': net.conv4.state_dict(),
        'bn4': net.bn4.state_dict(),
        'conv5': net.conv5.state_dict(),
        'bn5': net.bn5.state_dict()
    }

    torch.save(conv_layers_state_dict, 'models/conv_layers_cuda.pth')
