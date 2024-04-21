import argparse
import matplotlib.pyplot as plt
import multiprocessing
import time
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm

IMAGE_CLASS_NAMES = ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic']

def parse_args():
    parser = argparse.ArgumentParser(description='Train VGG16 on Garbage classification dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (initial)')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--dataset_dir', type=str, default='dataset/Garbage classification', help='Dataset directory')
    parser.add_argument('--model_save_path', type=str, default='vgg16.pth', help='Path to save the model')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()


def load_data(dataset_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = ImageFolder(dataset_dir, transform = transformations)

    # split the dataset into training, validation and test set
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_dl = DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_set, batch_size, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_set, batch_size, num_workers=4, pin_memory=True)
    return train_dl, val_dl, test_dl


class VGG16(nn.Module):
    def __init__(self, num_classes: int):
        super(VGG16, self).__init__()
        network = models.vgg16(weights='DEFAULT')   # load the pre-trained VGG model
        self.features = network.features
        self.avgpool = network.avgpool
        self.classifier = network.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)   # replace the last layer

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))


def train(model: nn.Module, data_loader: DataLoader, loss_fn, optimizer: torch.optim.Optimizer, device) -> float:
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0

    for _, (images, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader)


def validate(model: nn.Module, data_loader: DataLoader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(data_loader), correct / total


def main():
    args = parse_args()
    print(f'Arguments: {args}')

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')

    train_dl, val_dl, test_dl = load_data(args.dataset_dir, args.batch_size)

    model = VGG16(len(IMAGE_CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.num_epochs // 3, gamma=0.5)

    # train the model
    train_losses = []
    val_accuracy = []
    best_model = {'epoch': 0, 'val_acc': 0, 'model': None}
    for epoch in range(1, args.num_epochs + 1):
        begin_time = time.time()
        train_loss = train(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_dl, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_accuracy.append(val_acc)
        if val_acc > best_model['val_acc']:
            best_model = {'epoch': epoch, 'val_acc': val_acc, 'model': model.state_dict()}
        print(f'Epoch {epoch}/{args.num_epochs} ({time.time() - begin_time:.2f}s), Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    print(f'Best model at epoch {best_model["epoch"]}, Val Acc: {best_model["val_acc"]:.4f}')

    # plot the training loss and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # test the model
    model.load_state_dict(best_model['model'])  # load the best model
    test_loss, test_acc = validate(model, test_dl, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # save the model
    model_save_path = 'vgg16.pth'
    model.save_model(model_save_path)
    print(f"Model saved as '{model_save_path}'")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
