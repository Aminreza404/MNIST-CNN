import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torchvision 
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm.auto import tqdm

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))]) 

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(testset, batch_size=64, shuffle=False)

print("MNIST dataset loaded!")
print("=====================================================")


print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}")

print("Visualization...") 

dataiter = iter(trainloader) 
images, labels = next(dataiter) 

plt.figure(figsize=(8, 5)) 
for i in range(8): 
    plt.subplot(2, 4, i+1) 
    plt.imshow(images[i].squeeze(), cmap = 'gray') 
    plt.title(f"Label: {labels[i].item()}")
plt.show() 

class MNIST_CNN(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) 

        self.fc1 = nn.Linear(64*7*7, 128) 
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x): 
        x = torch.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = torch.relu(self.conv2(x)) 
        x = self.pool(x) 
        x = x.view(-1, 64*7*7) 
        x = torch.relu(self.fc1(x)) 
        x = self.dropout(x) 
        x = self.fc2(x) 

        return x 
    
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 
model = MNIST_CNN().to(device) 
model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

num_epoch = 15 
train_losses, test_losses = [], [] 
train_acc, test_acc = [], [] 

for epoch in tqdm(range(num_epoch)): 
    model.train() 
    running_loss = 0 
    correct_train = 0 
    total_train = 0 
    for images, labels in trainloader: 
        images, labels = images.to(device), labels.to(device) 
        optimizer.zero_grad() 
        outputs = model(images) 
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 


        running_loss += loss.item() 
        _, preds = torch.max(outputs, 1) 
        total_train += labels.size(0) 
        correct_train += (preds == labels).sum().item() 

    train_losses.append(running_loss / len(trainloader))
    train_acc.append(100 * correct_train / total_train) 

    model.eval()
    running_loss_test = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss_test += loss.item()
            _, preds = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (preds == labels).sum().item()
    
    test_losses.append(running_loss_test / len(testloader))
    test_acc.append(100 * correct_test / total_test)
    
    print(f"Epoch {epoch+1}/{num_epoch} | "
          f"Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f} | "
          f"Train Acc: {train_acc[-1]:.2f}%, Test Acc: {test_acc[-1]:.2f}%")

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(train_acc, label='Train Acc')
plt.plot(test_acc, label='Test Acc')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.show()

dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(12,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.title(f"Pred: {preds[i].item()}, True: {labels[i].item()}")
    plt.axis('off')
plt.show()



