# Gesture Recognition with Spiking Neural Networks & ABC Optimization

Gesture recognition using Spiking Neural Networks (SNNs) trained on DVS event-camera data. Model hyperparameters are optimized with the **Artificial Bee Colony (ABC)** swarm intelligence algorithm.

---

## How It Works

The model receives an event stream from a DVS camera, converts it into a series of binary frames (positive/negative polarity channels), and passes them through a convolutional SNN. Classification is based on the total spike count at the output layer accumulated over all time steps.

To find optimal `beta` (membrane potential decay constant) and `learning rate`, the **ABC** algorithm trains multiple model instances with different hyperparameter configurations and selects the best one.

---

## Project Structure

```
.
├── data/
│   └── EMG_DVS/                        # Dataset: DVS + EMG spikes (.pkl)
├── src/
│   ├── core/
│   │   └── model/
│   │       └── model.py                # GestureSNN architecture
│   ├── optimization/
│   │   ├── abc_optimizer.py            # Artificial Bee Colony
│   │   ├── aco_optimizer.py            # Ant Colony Optimization
│   │   └── pso_optimizer.py            # Particle Swarm Optimization
│   └── pipeline/
│       ├── dataloader/
│       │   ├── dvs_gesture.py          # DVS128 Gesture loader (tonic)
│       │   └── emg_pkl.py              # EMG+DVS loader from .pkl
│       ├── train.py                    # Baseline training
│       ├── train_optimized_model.py    # Training with ABC-found parameters
│       └── export_to_nir.py           # Export to NIR format
├── snn.ipynb                           # Jupyter notebook for experiments
├── best_emg_model.pth                  # Saved weights of the best baseline model
├── best_params.json                    # Best hyperparameters found by ABC
├── requirements.txt
└── README.md
```

---

## Model: GestureSNN

`src/core/model/model.py`

A two-layer convolutional SNN with Leaky Integrate-and-Fire (LIF) neurons:

```
Conv2d(2→16, 4×4, stride=4) → LIF → AvgPool2d(2)
Conv2d(16→32, 3×3, pad=1)   → LIF → AvgPool2d(2)
Flatten → Linear(32×8×8 → num_classes) → LIF
```

- **Input:** `[Time, Batch, 2, 128, 128]` — 2 channels (positive/negative polarity)
- **Output:** spikes at each time step; classification via `sum(dim=0)`
- **Defaults:** `beta=0.95`, `num_classes=11` (DVS Gesture) or `5` (EMG/DVS dataset)
- **Surrogate gradient:** `fast_sigmoid` for backpropagation through spikes

---

## Datasets

### DVS128 Gesture Dataset (IBM)
11 gesture classes recorded with a DVS128 event camera.  
Loaded via the `tonic` library:

```python
from src.pipeline.dataloader.dvs_gesture import get_dvs_gesture_loaders
train_loader, test_loader = get_dvs_gesture_loaders(batch_size=16, data_path='./data')
```

The event stream is converted into frames with a `50 ms` time window.

### EMG+DVS Dataset (.pkl)
File `relax21_cropped_dvs_emg_spikes.pkl` contains the following fields:
- `dvs` — list of events `(x, y, t, polarity)`
- `y` — class labels (5 gestures)
- `sub` / `subject` — subject ID

Split: subjects 1–16 for training, 17–21 for testing.

```python
from src.pipeline.dataloader.emg_pkl import get_emg_pkl_loaders
train_loader, test_loader = get_emg_pkl_loaders(
    pkl_path="data/EMG_DVS/relax21_cropped_dvs_emg_spikes.pkl",
    batch_size=8,
    train_subjects=list(range(1, 17)),
    test_subjects=list(range(17, 22)),
    num_frames=20
)
```

Each event stream is converted into `num_frames=20` binary frames of shape `[2, 128, 128]`.

---

## Installation

```bash
git clone https://github.com/TereshchenkoMisha/Gesture-recognition.git
cd Gesture-recognition
pip install -r requirements.txt
```

Key dependencies: `torch`, `snntorch`, `tonic`, `nir`.

---

## Usage

### 1. Baseline training

```bash
python src/pipeline/train.py
```

Parameters are set inside the file:
- `epochs = 50`, `batch_size = 8`, `lr = 5e-4`
- Best model is saved to `best_emg_model.pth`

### 2. Hyperparameter optimization (ABC)

```bash
python src/optimization/abc_optimizer.py
```

The algorithm searches over `beta ∈ [0.5, 0.99]` and `lr ∈ [0.0001, 0.002]`.  
Results are saved to `best_params.json`.

Default configuration: `num_bees=6`, `iter_max=3`, `limit=2`.

### 3. Training with optimized parameters

```bash
python src/pipeline/train_optimized_model.py
```

Reads `best_params.json`, trains for 50 epochs with `ReduceLROnPlateau` scheduler.  
Best weights are saved to `best_optimized_model.pth`.

### 4. Export to NIR

```bash
python src/pipeline/export_to_nir.py
```

Exports the trained model (`best_emg_model.pth`) to [NIR](https://github.com/neuromorphs/NIR) (Neuromorphic Intermediate Representation) format for deployment on neuromorphic hardware.

---

## ABC Algorithm

`src/optimization/abc_optimizer.py`

Implements the classic three-phase ABC:

| Phase | Description |
|-------|-------------|
| **Employed bees** | Each bee modifies its food source (hyperparameter config) and evaluates fitness |
| **Onlooker bees** | Select sources proportionally to fitness and explore their neighborhood |
| **Scout bees** | Replace exhausted sources (`trials > limit`) with random new ones |

Fitness function: validation error (`1 - accuracy`) after 3 training epochs.

---

## Team

- **Saida Musaeva** — SNN architecture, ABC algorithm
- **Mikhail Tereshchenko** — training pipeline, hyperparameter optimization
- **Anna Tikhonova** — data preparation, visualization
