import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import time

# Define transformations, load dataset, and split into train and test
def setup_data():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.RandomResizedCrop(224)
    ])
    trainpath = r'CS171Project\\train'
    testpath = r'CS171Project\\test'
    trainset = ImageFolder(root=trainpath, transform=transform)
    testset = ImageFolder(root=testpath, transform=transform)
    # trainset, testset = torch.utils.data.random_split(dataset, [...])
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader

# Load and modify the model
def setup_model():
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

# Training function
def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    since = time.time()
    for epoch in range(epochs):
        for inputs,labels in trainloader:
            outputs = model(inputs)
            loss = criterion(outputs,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    print('Finished Training')
    time_elasped = time.time() - since
    print(time_elasped)

# Evaluation function
def evaluate_model(model, testloader):
    model.eval()
    ...
    # print(f'Accuracy: {acc}, F1-Score: {f1}')
def main():

    trainloader, testloader = setup_data()
    model = setup_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, trainloader, criterion, optimizer)
    evaluate_model(model, testloader)

if __name__ == "__main__":
    main()
