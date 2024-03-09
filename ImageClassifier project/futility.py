import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import json
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable

class ModelArchitectures:
    CONFIGS = {"vgg16": 25088, "densenet121": 1024}

def load_data(root_dir="./flowers"):
    with open('cat_to_name.json', 'r') as json_file:
        category_mapping = json.load(json_file)

    data_dir = root_dir
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define data transforms
    normalize_params = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Define the transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*normalize_params)
    ])

    test_transforms = valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*normalize_params)
    ])

    # Load the datasets with ImageFolder
    train_data = ImageFolder(train_dir, transform=train_transforms)
    test_data = ImageFolder(test_dir, transform=test_transforms)
    valid_data = ImageFolder(valid_dir, transform=valid_transforms)

    # Define the dataloaders
    def create_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Using the image datasets and the transforms, define the dataloaders
    batch_size = 64
    trainloader = create_dataloader(train_data, batch_size)
    testloader = create_dataloader(test_data, batch_size)
    validloader = create_dataloader(valid_data, batch_size)
