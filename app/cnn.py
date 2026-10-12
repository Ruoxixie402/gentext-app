"""Assignment 2 CNN and shared training/inference preprocessing."""

from torch import nn
from torchvision import transforms

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
ARCHITECTURE = "cifar10-cnn-64-v1"


class CIFAR10CNN(nn.Module):
    """RGB 64x64 -> 16x32x32 -> 32x16x16 -> 100 -> 10 logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(), nn.Linear(100, 10),
        )

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
            raise ValueError("Expected a batch of RGB images with shape (N, 3, 64, 64).")
        return self.classifier(self.features(images))


def image_transform():
    # CIFAR-10 images are 32x32; the assignment architecture requires 64x64.
    return transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
