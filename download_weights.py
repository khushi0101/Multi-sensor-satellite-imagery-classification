from pathlib import Path
import torch
from torchvision.models import resnet18, ResNet18_Weights

Path("models").mkdir(exist_ok=True)
net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
torch.save(net.state_dict(), "models/resnet18_imagenet.pth")
print("saved")