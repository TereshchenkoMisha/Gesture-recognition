import sys
import os
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(current_dir)

from src.core.model.model import GestureSNN, calc_accuracy
from src.pipeline.dataloader.emg_pkl import get_emg_pkl_loaders

# Loading of best parameters from ABC
with open("best_params.json", "r") as f:
    best_params = json.load(f)
BEST_BETA = best_params["beta"]
BEST_LR = best_params["lr"]

# Settings
PKL_PATH = "data/EMG_DVS/relax21_cropped_dvs_emg_spikes.pkl"
BATCH_SIZE = 8
EPOCHS = 50
WEIGHT_DECAY = 1e-5
NUM_FRAMES = 20

# Data: tarining set – subjects 1-16, testing set – 17-21
train_subjects = list(range(1, 17))
test_subjects = list(range(17, 22))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Using best params: beta={BEST_BETA:.4f}, lr={BEST_LR:.5f}")

# Data loading
train_loader, test_loader = get_emg_pkl_loaders(
    pkl_path=PKL_PATH,
    batch_size=BATCH_SIZE,
    train_subjects=train_subjects,
    test_subjects=test_subjects,
    num_frames=NUM_FRAMES
)
print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

# Creating model with standard parameters
model = GestureSNN(input_channels=2, num_classes=5)

# Manually install the optimized beta (and, optionally, threshold)
for module in model.modules():
    if hasattr(module, 'beta'):
        module.beta = torch.tensor(BEST_BETA, dtype=torch.float32, device=device)
    # ----------------for future modernozation-------------------
    # if hasattr(module, 'threshold'):
    #     module.threshold = torch.tensor(1.0, dtype=torch.float32, device=device)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=BEST_LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Learning
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        data = data.transpose(0, 1)  # (T, B, C, H, W)
        optimizer.zero_grad()
        spk_out = model(data)
        loss = loss_fn(spk_out.sum(dim=0), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, step {i}/{len(train_loader)}, loss={loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    val_acc = calc_accuracy(model, test_loader)
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:2d} | loss={avg_loss:.4f} | test_acc={val_acc*100:.2f}% | lr={current_lr:.2e}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_optimized_model.pth")
        print(f"  -> best model saved (acc={val_acc*100:.2f}%)")

print(f"\nFinal best test accuracy: {best_acc*100:.2f}%")