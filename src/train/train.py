import sys
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(current_dir)

from src.core.model.model import GestureSNN, calc_accuracy
from src.core.datasets.emg_pkl import get_emg_pkl_loaders

# Settings (parameters)
pkl_path = "data/EMG_DVS/relax21_cropped_dvs_emg_spikes.pkl"
batch_size = 8
epochs = 50
learning_rate = 5e-4
weight_decay = 1e-5
num_frames = 20   # fixed number of time steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_subjects = list(range(1, 17))
test_subjects = list(range(17, 22))

print(f"Устройство: {device}, batch_size={batch_size}, lr={learning_rate}, num_frames={num_frames}")

# Loading
train_loader, test_loader = get_emg_pkl_loaders(
    pkl_path=pkl_path,
    batch_size=batch_size,
    train_subjects=train_subjects,
    test_subjects=test_subjects,
    num_frames=num_frames
)

print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

# Info about data
sample_data, sample_targets = next(iter(train_loader))
print(f"Batch form: {sample_data.shape}")  # expected [8, 20, 2, 128, 128]
print(f"Targets: {sample_targets[:10]}")
print(f"Data min/max: {sample_data.min().item():.3f}/{sample_data.max().item():.3f}")

# Model
model = GestureSNN(input_channels=2, num_classes=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


# Training
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        data = data.transpose(0, 1)   # (B, T, C, H, W) -> (T, B, C, H, W)
        optimizer.zero_grad()
        spk_out = model(data)
        spk_count = spk_out.sum(dim=0)
        loss = loss_fn(spk_count, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, step {i}/{len(train_loader)}, loss={loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    val_acc = calc_accuracy(model, test_loader) if test_loader else 0.0
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:2d} | loss={avg_loss:.4f} | val_acc={val_acc*100:.2f}% | lr={current_lr:.2e}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_emg_model.pth")
        print(f"  -> best version saved (acc={val_acc*100:.2f}%)")
print(f"\nFinally! Best accuracy: {best_acc*100:.2f}%")