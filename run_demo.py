import os
import subprocess
import torch
import h5py
import numpy as np
from pathlib import Path
import sys
import shutil
import matplotlib.pyplot as plt

# Import model
sys.path.append(os.path.dirname(__file__))
from src.core.model.model import GestureSNN


def convert_video_to_dvs(video_path, output_dir="temp_dvs_demo"):
    """Convert video to DVS via v2e with optimal thresholds"""
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    cmd = [
        sys.executable, "v2e/v2e.py",
        "--input", video_path,
        "--output_folder", output_dir,
        "--output_height", "128",
        "--output_width", "128",

        "--dvs_exposure", "duration", "0.01",
        "--pos_thres", "0.25",
        "--neg_thres", "0.25",
        "--sigma_thres", "0.05",
        "--cutoff_hz", "0",

        "--auto_timestamp_resolution", "True",
        "--disable_slomo",
        "--dvs_h5", "events.h5"
    ]
    print(" Converting video to DVS")
    subprocess.run(cmd, check=True)
    events_path = Path(output_dir) / "events.h5"
    if not events_path.exists():
        raise FileNotFoundError("Failed to create events.h5")
    return events_path


def load_events(h5_path):
    """Load events with automatic column detection (t, x, y, p)"""
    with h5py.File(h5_path, "r") as f:
        events = f['events'][:]  # (N, 4)

    max_vals = np.max(events, axis=0)
    t_idx = np.argmax(max_vals)

    p_idx = 3

    coords_indices = [i for i in range(4) if i != t_idx and i != p_idx]
    x_idx, y_idx = coords_indices[0], coords_indices[1]

    t = events[:, t_idx].astype(np.float32)
    x = events[:, x_idx].astype(np.int32)
    y = events[:, y_idx].astype(np.int32)
    p_raw = events[:, p_idx].astype(np.int32)
    p = np.where(p_raw > 0, 1, 0)

    print(f"Columns detected: t={t_idx}, x={x_idx}, y={y_idx}, p={p_idx}")
    print(f"--- DEBUG ---")
    print(f"X range: {x.min()} - {x.max()}")
    print(f"Y range: {y.min()} - {y.max()}")
    print(f"T range: {t.min():.1f} - {t.max():.1f}")

    return x, y, t, p


def events_to_tensor(x, y, t, p, num_frames=20, size=128):
    """Convert events to tensor (T, 1, C, H, W)"""
    t_min, t_max = t.min(), t.max()
    duration = t_max - t_min
    if duration == 0: duration = 1.0

    tensor = np.zeros((num_frames, 2, size, size), dtype=np.float32)

    # Normalize time to range [0, num_frames-1]
    t_norm = (t - t_min) / duration
    bin_indices = (t_norm * (num_frames - 1)).astype(np.int32)

    for i in range(len(t)):
        xi, yi, bi, pi = x[i], y[i], bin_indices[i], p[i]
        if 0 <= xi < size and 0 <= yi < size:
            tensor[bi, pi, yi, xi] = 1.0

    final_tensor = torch.FloatTensor(tensor).unsqueeze(1)
    print(f"Tensor populated! Active points: {final_tensor.sum().item()}")
    return final_tensor


def visualize_inference(input_tensor):
    """Show what the neural network sees (sum of all frames)"""
    summed = torch.sum(input_tensor[:, 0, 0], dim=0).cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(summed, cmap='hot')
    plt.title("Input gesture visualization")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds then continue
    plt.close()


def main():
    video_path = input("Enter video filename (e.g., like.mp4): ").strip()
    if not os.path.exists(video_path):
        print(f"File {video_path} not found!")
        return

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # Load model
    try:
        model = GestureSNN(input_channels=2, num_classes=5).to(device)
        model.load_state_dict(torch.load("best_emg_model.pth", map_location=device))
        model.eval()
        print(" Model loaded.")
    except Exception as e:
        print(f" Model loading error: {e}")
        return

    try:
        # 1. Conversion
        h5_path = convert_video_to_dvs(video_path)

        # 2. Load and fix columns
        x, y, t, p = load_events(h5_path)

        # 3. Convert to tensor
        input_tensor = events_to_tensor(x, y, t, p).to(device)

        # 4. Visual check
        visualize_inference(input_tensor)

        # 5. Inference
        with torch.no_grad():
            spk_out = model(input_tensor)
            # spk_out shape: (T, Batch, Classes)
            spk_count = spk_out.sum(dim=0)
            pred = torch.argmax(spk_count, dim=1).item()

            print(f"Raw spikes per class: {spk_count.cpu().numpy()}")

        emoji_map = {0: "✊", 1: "👆", 2: "☝️", 3: "👍", 4: "🤘"}
        print(f"Result: {emoji_map.get(pred)} (class {pred})")

    except Exception as e:
        print(f" Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()