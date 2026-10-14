"""Train on dataset/{left,right,straight}/*.jpg, then export to ONNX."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import LFRNet, IMG_SIZE

DATA_DIR = "dataset"
EPOCHS = 15
BATCH = 32
LR = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

norm = transforms.Normalize([0.5] * 3, [0.5] * 3)  # must match infer_pi.py
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
    norm,
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    norm,
])

full = datasets.ImageFolder(DATA_DIR)
print("classes:", full.classes)
n_val = int(0.2 * len(full))
train_idx, val_idx = random_split(range(len(full)), [len(full) - n_val, n_val],
                                  generator=torch.Generator().manual_seed(0))
train_ds = torch.utils.data.Subset(datasets.ImageFolder(DATA_DIR, train_tf), train_idx.indices)
val_ds = torch.utils.data.Subset(datasets.ImageFolder(DATA_DIR, val_tf), val_idx.indices)
train_dl = DataLoader(train_ds, BATCH, shuffle=True)
val_dl = DataLoader(val_ds, BATCH)

model = LFRNet(len(full.classes)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
best = 0.0

for ep in range(EPOCHS):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_dl:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    acc = correct / len(val_ds)
    print(f"epoch {ep + 1}/{EPOCHS}  val_acc={acc:.3f}")
    if acc > best:
        best = acc
        torch.save(model.state_dict(), "lfr_best.pt")

# ---- export best weights to ONNX ----
model.load_state_dict(torch.load("lfr_best.pt", map_location=device))
model.eval().cpu()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(model, dummy, "lfr.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
print("best val_acc:", best, "-> saved lfr.onnx")
