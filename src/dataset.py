import torch
from torchvision import datasets, transforms

def get_data_loader(batch_size = 32):
    transform = transforms.Compose([
        transforms.ToTensor()   # convert it to 0 and 1
    ])

    train_data = datasets.MNIST(root = './data', train = True, transform = transform , download = True)
    test_data = datasets.MNIST(root= './data', train = False, transform = transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)
    return train_loader, test_loader