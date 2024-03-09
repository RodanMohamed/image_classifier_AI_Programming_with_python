import argparse
import torch
from torch import nn, optim
import futility as fu
import fmodel as fm
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import numpy as np
import os, random
import matplotlib
import matplotlib.pyplot as plt
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser for train.py')
    parser.add_argument('data_dir', action="store", default="./flowers/")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', action="store", default="vgg16")
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
    parser.add_argument('--epochs', action="store", default=3, type=int)
    parser.add_argument('--dropout', action="store", type=float, default=0.2)
    parser.add_argument('--gpu', action="store", default="gpu")

    return parser.parse_args()

def set_device(power):
    if torch.cuda.is_available() and power == 'gpu':
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")
def main():
    args = parse_arguments()
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    gpu = args.gpu
    epochs = args.epochs
    dropout = args.dropout

    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu == 'gpu' else "cpu")

    # Load data
    trainloader, validloader, testloader, train_data = fu.load_data(data_dir)

    # Setup model and criterion
    model, criterion = fm.setup_network(arch, dropout, hidden_units, learning_rate, gpu)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Train Model
    steps = 0
    running_loss = 0
    print_every = 5
    print("--Training starting--")

    num_epochs = 3
    print_interval = 5
    total_steps = 0
    loss_history = []

    for epoch in range(num_epochs):
    running_loss = 0

    for inputs, labels in trainloader:
        total_steps += 1
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if total_steps % print_interval == 0:
            model.eval()
            validation_loss = 0
            accuracy = 0

            with torch.no_grad():
                for val_inputs, val_labels in validloader:
                    val_inputs, val_labels = val_inputs.to('cuda'), val_labels.to('cuda')

                    log_ps = model(val_inputs)
                    batch_loss = criterion(log_ps, val_labels)
                    validation_loss += batch_loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == val_labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Training Loss: {running_loss / print_interval:.3f}, "
                  f"Validation Loss: {validation_loss / len(validloader):.3f}, "
                  f"Validation Accuracy: {accuracy / len(validloader):.3f}")

            running_loss = 0
            model.train()

    
    model.class_to_idx = train_data.class_to_idx
    torch.save({
        'structure': arch,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'no_of_epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }, save_dir)

    print("Saved checkpoint!")

if __name__ == "__main__":
    main()
