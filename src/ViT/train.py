import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from model.vit import ViT
from torch.optim import Adam
from tqdm import tqdm


def test(model, device, loader, loss, patches):
    model.eval()
    test_loss = 0
    correct = 0
    total_imgs = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = patches(data)
            preds = model(data)
            test_loss += loss(preds, target).item()
            correct += torch.sum(torch.argmax(preds, dim=1) == target).item()
            total_imgs += len(data)
    test_loss /= total_imgs
    acc = correct/ total_imgs
    return test_loss, 100. * acc

def main():
    patch_res = 4
    num_epochs = 5
    bs = 100

    mnist_trainset = MNIST('./data', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ]))
    mnist_valset = MNIST('./data', download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ]))

    def create_patches(batch):
        bs, channels, height, width = batch.shape
        num_patches = (height * width)//(patch_res ** 2)
        patch_size = (patch_res ** 2) * channels
        return torch.reshape(batch, (bs, num_patches, patch_size))

    trainloader = DataLoader(mnist_trainset, batch_size=bs, shuffle=True)
    valloader = DataLoader(mnist_valset, batch_size=bs, shuffle=False)

    model = ViT(d_model= 64, n_patches= 7, n_encoders=2, n_heads=8, patch_res=patch_res)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print(f"Using device: {device}")
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.03)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        total_imgs_tr = 0
        print(f"Epoch: {epoch}")
        for img, labels in tqdm(trainloader, total=len(list(trainloader))):
            img = img.to(device)
            labels = labels.to(device)
            img = create_patches(img)
            optimizer.zero_grad()

            pred = model(img)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            total_imgs_tr += len(img)

            epoch_loss += loss.item()
        avg_loss = epoch_loss / total_imgs_tr
        test_loss, acc = test(model, device, valloader, criterion, create_patches)
        print(f"Training loss: {avg_loss}, Validation loss: {test_loss}, Validation accuracy: {acc}")


if __name__ == "__main__":
    main()