import argparse
import torch
import futility as fu
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import numpy as np
import os, random
import matplotlib
import matplotlib.pyplot as plt
import json
from torch.autograd import Variable
import fmodel

def get_command_line_arguments():
    parser = argparse.ArgumentParser(description='Image Classifier Predictions')
    parser.add_argument('image_path', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type=str)
    parser.add_argument('--data_dir', action="store", dest="data_dir", default="./flowers/", help='Path to the data directory')
    parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='Top K most likely classes')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='Path to the category names mapping file')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu", help='Use GPU if available')

    return parser.parse_args()

def main():
    args = get_command_line_arguments()

    image_path = args.image_path
    top_k = args.top_k
    device = args.gpu
    checkpoint_path = args.checkpoint

    model = fmodel.load_checkpoint(checkpoint_path)

    with open(args.category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)

    probabilities = fmodel.predict(image_path, model, top_k, device)
    probability = np.array(probabilities[0][0])
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]

    for label, prob in zip(labels, probability):
        print(f"Class: {label}, Probability: {prob:.4f}")

    print("Prediction Completed!")

if __name__ == "__main__":
    main()
