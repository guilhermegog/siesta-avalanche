import torch
import torchvision
from torchvision import transforms
from torch import nn
from torchvision.models import mobilenet_v3_small
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import v2
import random
import time
from siesta_mobilenet import mobilenet_v3_large, CosineLinear


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_class_names(file_path, num_classes=100):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f]
    return class_names[:num_classes]


class FilteredImageNet(Dataset):
    def __init__(self, full_dataset, selected_indices, remapped_labels):
        self.full_dataset = full_dataset
        self.selected_indices = selected_indices
        self.remapped_labels = remapped_labels

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        original_idx = self.selected_indices[idx]
        image, _ = self.full_dataset[original_idx]
        return image, self.remapped_labels[idx]


def create_filtered_dataset(dataset, selected_class_ids, class_to_idx):
    filtered_indices = []
    remapped_labels = []
    for i, (_, class_id) in enumerate(dataset.samples):
        if class_id in selected_class_ids:
            filtered_indices.append(i)
            remapped_labels.append(class_to_idx[class_id])

    return FilteredImageNet(dataset, filtered_indices, remapped_labels)


def get_imagenet_subset(num_classes=100, batch_size=128):
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))
    ])

    t0 = time.time()
    full_dataset = torchvision.datasets.ImageNet(
        root='/space/gguedes/datasets/imagenet1k/', split='train')
    test_dataset = torchvision.datasets.ImageNet(
        root='/space/gguedes/datasets/imagenet1k/', split='val')
    tt = time.time()
    print("Dataset is loaded :D: ", tt-t0)

    # Filter dataset to only include desired classes and remap labels
    selected_class_syn = read_class_names(
        'model_v2/imagenet_class_order.txt', num_classes=100)
    print(selected_class_syn[:10])
    syn_to_idx = {}

    for syn in selected_class_syn:
        syn_to_idx[syn] = full_dataset.wnid_to_idx[syn]

    print(syn_to_idx.keys())
    selected_class_ids = [syn_to_idx[name]
                          for name in selected_class_syn]
    class_to_idx = {class_id: idx for idx,
                    class_id in enumerate(selected_class_ids)}

    full_dataset = torchvision.datasets.ImageNet(
        root='/space/gguedes/datasets/imagenet1k/', split='train', transform=transform)
    test_dataset = torchvision.datasets.ImageNet(
        root='/space/gguedes/datasets/imagenet1k/', split='val', transform=transform)

    filtered_dataset = create_filtered_dataset(
        full_dataset, selected_class_ids, class_to_idx)
    filtered_test_dataset = create_filtered_dataset(
        test_dataset, selected_class_ids, class_to_idx)
    train_loader = DataLoader(
        filtered_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(
        filtered_test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    t1 = time.time()
    print("Total Loading: ", t1-t0)
    return train_loader, test_loader, selected_class_ids


def adapt_mbnet(num_classes=100):
    model = mobilenet_v3_large()

    model.classifier[3] = CosineLinear(
        model.classifier[3].in_features, num_classes)
    print(model.features)
    return model


def load_partial_model(model, checkpoint_path, num_frozen_layers=8):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()

    # Filter out layers after the 8th layer
    pretrained_dict = {k: v for k, v in checkpoint.items(
    ) if k in model_dict and 'features.' in k and int(k.split('.')[1]) < num_frozen_layers}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)

    # Freeze the first 8 layers
    for name, param in model.named_parameters():
        if 'features.' in name and int(name.split('.')[1]) < num_frozen_layers:
            param.requires_grad = False

    return model


def save_partial_model(model, save_path, start_layer=8):
    state_dict = model.state_dict()
    save_dict = {k: v for k, v in state_dict.items(
    ) if 'features.' not in k or int(k.split('.')[1]) >= start_layer}
    torch.save(save_dict, save_path)
