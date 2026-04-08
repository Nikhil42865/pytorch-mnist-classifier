import torch
import torch.nn as nn
from dataset import get_data_loader
from model import neuralNet

train_loader , test_loader = get_data_loader(batch_size = 32)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = neuralNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

train_losses = []
train_accuracies = []

for epoch in range(5):
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch {epoch}, Avg Loss: {epoch_loss:.4f}")