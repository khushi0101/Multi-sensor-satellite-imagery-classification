from pathlib import Path
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import json


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path("/Users/khushiagrawal/Documents/Khushi/Galaxeye assignment/Multi-sensor satellite image classification/be-mlsys-assignment-dataset")
DATA_DIR = BASE_DIR / "candidate_tiles"
WEIGHTS_PATH = SCRIPT_DIR / "models" / "resnet18_imagenet.pth"
X_PATH = SCRIPT_DIR / "X.npy"
Y_PATH = SCRIPT_DIR / "y.npy"
CLASSES_PATH = SCRIPT_DIR / "classes.json"

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_feature_extractor():
    net = models.resnet18(weights=None)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    net.fc = torch.nn.Identity()
    net.eval()
    return net

# net = models.resnet18(weights=None)
# net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))

# # 2. Remove the last layer (1000 ImageNet classes) -> outputs 512 features
# net.fc = torch.nn.Identity()

# # 3. Inference mode
# net.eval()

# 4. Preprocessing (must match what ResNet was trained on)
def load_image(src):
    
    return Image.open(src)

def collect_path_labels():
    
    classes = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())

    paths, labels = [], []
    for idx, cls in enumerate(classes):
        for p in sorted((DATA_DIR / cls).glob("*.png")):
            paths.append(p)
            labels.append(idx)

    return paths, labels, classes

def extract_features(net, paths, batch_size=64):
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            tensors = [preprocess(load_image(p)) for p in batch_paths]
            batch = torch.stack(tensors)          # (B, 3, 224, 224)
            feats = net(batch)                    # (B, 512)
            all_feats.append(feats.numpy())
            print(f"{i + len(batch_paths)}/{len(paths)}")
    return np.concatenate(all_feats)              # (N, 512)


if __name__ == "__main__":
    paths, labels, classes = collect_path_labels()
    net = load_feature_extractor()
    X = extract_features(net, paths)
    y = np.array(labels)
    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    CLASSES_PATH.write_text(json.dumps(classes))
    print(X.shape, y.shape)
    print(f"Saved X {X.shape}, y {y.shape}, classes -> {SCRIPT_DIR}")
