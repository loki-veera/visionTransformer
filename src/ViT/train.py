import torch
import torch.nn as nn
from torch import utils
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from .model.vit import ViT


def create_patches(batch: torch.Tensor, patch_res: int = 4) -> torch.Tensor:
    """Compute the patches of the batch.

    Args:
        batch (torch.tensor): Tensor containing the batch of images.
        patch_res (int, optional): Resolution of each patch. Defaults to 4.

    Returns:
        torch.tensor: Tensor containing the patches of each image.
    """
    bs, channels, height, width = batch.shape
    num_patches = (height * width) // (patch_res**2)
    patch_size = (patch_res**2) * channels
    return torch.reshape(batch, (bs, num_patches, patch_size))


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: utils.data.DataLoader,
    loss: nn.Module,
) -> tuple:
    """Compute the test accuracies.

    Args:
        model (nn.Module): Transformer model.
        device (torch.device): torch device either cuda or cpu.
        loader (utils.DataLoader): Test loader.
        loss (nn.Module): Loss function.

    Returns:
        tuple: tuple containing testloss and accuracy
    """
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total_imgs = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = create_patches(data)
            preds = model(data)
            test_loss += loss(preds, target).item()
            correct += torch.sum(torch.argmax(preds, dim=1) == target).item()
            total_imgs += len(data)
    test_loss /= total_imgs
    acc = correct / total_imgs
    return test_loss, 100.0 * acc


def train(
    model: nn.Module,
    device: torch.device,
    loader: utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Training loop.

    Args:
        model (nn.Module): Transformer model.
        device (torch.device): torch device either cuda or cpu.
        loader (DataLoader): Train loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim): Optimizer function.

    Returns:
        float: Loss value for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_imgs = 0
    for img, labels in tqdm(loader, total=len(list(loader))):
        img = img.to(device)
        labels = labels.to(device)
        img = create_patches(img)
        optimizer.zero_grad()

        pred = model(img)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        total_imgs += len(img)

        total_loss += loss.item()
    avg_loss = total_loss / total_imgs
    return avg_loss


def main():
    """Training and evaluation loop."""
    num_epochs = 30
    bs = 100
    patch_res = 4  # Resolution of each patch
    torch.manual_seed(21)

    mnist_trainset = MNIST(
        "./data",
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    mnist_valset = MNIST(
        "./data",
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    trainloader = DataLoader(mnist_trainset, batch_size=bs, shuffle=True)
    valloader = DataLoader(mnist_valset, batch_size=bs, shuffle=False)

    model = ViT(d_model=64, n_patches=7, n_encoders=2, n_heads=8, patch_res=patch_res)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print(f"Using device: {device}")
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.003)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        train_loss = train(model, device, trainloader, criterion, optimizer)
        val_loss, acc = evaluate(model, device, valloader, criterion)
        print(
            f"Training loss: {train_loss}, Validation loss: {val_loss}, Validation accuracy: {acc}"
        )


if __name__ == "__main__":
    main()
