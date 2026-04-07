import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class EMG_PKL_Dataset(Dataset):
    def __init__(self, pkl_path, num_frames=20, sensor_size=(128,128), subjects=None):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print("Доступные ключи в .pkl:", data.keys())
        self.dvs_list = data['dvs']
        self.labels = data['y']
        self.subjects = data.get('sub', data.get('subject', None))
        if subjects is not None and self.subjects is not None:
            idx = [i for i, s in enumerate(self.subjects) if s in subjects]
            self.dvs_list = [self.dvs_list[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]
        self.num_frames = num_frames
        self.sensor_size = sensor_size
        print(f"Загружено {len(self.dvs_list)} образцов, num_frames={num_frames}")

    def __len__(self):
        return len(self.dvs_list)

    def events_to_frames(self, events):
        xs = events[0].astype(int)
        ys = events[1].astype(int)
        ts = events[2]
        ps = events[3].astype(int)
        t_min, t_max = ts.min(), ts.max()
        if t_max == t_min:
            t_max = t_min + 1
        frames = []
        for i in range(self.num_frames):
            start = t_min + i * (t_max - t_min) / self.num_frames
            end = t_min + (i+1) * (t_max - t_min) / self.num_frames
            mask = (ts >= start) & (ts < end)
            pos = np.zeros(self.sensor_size, dtype=np.float32)
            neg = np.zeros(self.sensor_size, dtype=np.float32)
            for x, y, p in zip(xs[mask], ys[mask], ps[mask]):
                if 0 <= x < self.sensor_size[0] and 0 <= y < self.sensor_size[1]:
                    if p == 1:
                        pos[y, x] = 1
                    else:
                        neg[y, x] = 1
            frames.append(np.stack([pos, neg], axis=0))
        return np.array(frames)

    def __getitem__(self, idx):
        events = self.dvs_list[idx]
        frames = self.events_to_frames(events)
        frames = torch.tensor(frames, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return frames, label

def custom_collate(batch):
    frames, labels = zip(*batch)
    max_T = max(f.shape[0] for f in frames)
    C, H, W = frames[0].shape[1], frames[0].shape[2], frames[0].shape[3]
    padded = []
    for f in frames:
        T = f.shape[0]
        if T < max_T:
            pad = torch.zeros((max_T - T, C, H, W))
            padded.append(torch.cat([f, pad], dim=0))
        else:
            padded.append(f)
    return torch.stack(padded, dim=0), torch.tensor(labels)

def get_emg_pkl_loaders(pkl_path, batch_size, train_subjects=None, test_subjects=None, num_frames=20):
    train_ds = EMG_PKL_Dataset(pkl_path, num_frames=num_frames, subjects=train_subjects)
    test_ds = EMG_PKL_Dataset(pkl_path, num_frames=num_frames, subjects=test_subjects) if test_subjects else None
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, collate_fn=custom_collate) if test_ds else None
    return train_loader, test_loader