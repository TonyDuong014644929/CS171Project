import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

def setup_data():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    path = Path('.')
    trainpath = path / 'train'
    testpath = path / 'test'
    dataset = ImageFolder(root=trainpath, transform=transform)
    testset = ImageFolder(root=testpath, transform=transform)
    
    # Splitting the dataset into train and validation
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))
    
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    trainset = torch.utils.data.Subset(dataset, train_idx)
    valset = torch.utils.data.Subset(dataset, val_idx)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    return trainloader, valloader, testloader

def setup_model(num_classes=2):
    model = mobilenet_v2(pretrained=True)
    # Change the last layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def train_model(model, trainloader, valloader, criterion, optimizer, epochs=10, device='cpu'):
    model.to(device)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            print(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        val_acc = evaluate_model(model, valloader, device)
        print(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f} Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    print('Finished Training')

def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader, valloader, testloader = setup_data()
    model = setup_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, trainloader, valloader, criterion, optimizer, device=device)
    model.load_state_dict(torch.load('best_model.pth'))
    test_acc = evaluate_model(model, testloader, device)
    print(f'Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()
