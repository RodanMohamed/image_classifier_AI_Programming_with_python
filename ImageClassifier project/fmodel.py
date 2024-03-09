import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import json
import cv2
from PIL import Image
import futility

def create_custom_model(structure='vgg16', dropout_rate=0.1, hidden_units=4096, learning_rate=0.001, use_gpu=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    if structure == 'vgg16':
        base_model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        base_model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Unsupported model structure: {structure}")

    for param in base_model.parameters():
        param.requires_grad = False

    custom_classifier = nn.Sequential(
        nn.Linear(base_model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    base_model.classifier = custom_classifier
    custom_model = base_model.to(device)

    custom_criterion = nn.NLLLoss()
    custom_optimizer = optim.Adam(custom_model.classifier.parameters(), learning_rate)

    return custom_model, custom_criterion, custom_optimizer

def save_checkpoint(train_data, model=0, optimizer=0, path='checkpoint.pth', structure='vgg16', hidden_units=4096, dropout=0.3, lr=0.001, epochs=1):
    checkpoint = {
        'input_size': 25088,
        'output_size': 102,
        'structure': structure,
        'learning_rate': lr,
        'classifier': model.classifier,
        'epochs': epochs,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx
    }
    torch.save(checkpoint, path)

def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)

    learning_rate = checkpoint['learning_rate']
    hidden_units = checkpoint['classifier'][0].out_features
    dropout = checkpoint['classifier'][2].p
    epochs = checkpoint['epochs']
    structure = checkpoint['structure']

    model, _, _ = create_custom_model(structure, dropout, hidden_units, learning_rate)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def predict(image_path, model, top_k=5):
    model.eval()

    with torch.no_grad():
        # Process the image and convert to a PyTorch tensor
        image = process_image(image_path)
        image_tensor = image.unsqueeze(0).to('cuda')

        # Perform the forward pass and get the probabilities
        log_probs = model(image_tensor)

        # Calculate the probabilities and top classes
        probabilities = torch.exp(log_probs).cpu().numpy()[0]
        top_classes = np.argsort(-probabilities)[:top_k]

    return probabilities[top_classes], top_classes

def process_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))

    # Center crop the image
    center_crop_size = 224
    height, width = img.shape[:2]
    start_y = (height - center_crop_size) // 2
    start_x = (width - center_crop_size) // 2
    img = img[start_y:start_y+center_crop_size, start_x:start_x+center_crop_size]

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize the pixel values
    img = img / 255.0
    # Standardize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    # Transpose to PyTorch format (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    # Convert to PyTorch tensor
    img = torch.tensor(img, dtype=torch.float32)

    return img
