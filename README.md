
# Gesture Recognition with Spiking Neural Networks & ABC Optimization

## Project Overview

This project focuses on **gesture recognition** using **Spiking Neural Networks (SNNs)**. SNNs are biologically‑plausible neural networks that process temporal information efficiently, making them ideal for dynamic gesture data from event‑based cameras or video streams.  

To maximise accuracy and generalisation, we optimise the SNN hyperparameters with the **Artificial Bee Colony (ABC)** algorithm – a swarm‑intelligence metaheuristic.

### Key Features

- SNN training (e.g., using `snnTorch` or a custom PyTorch implementation)
- Hyperparameter optimisation with ABC (thresholds, time constants, learning rate, number of spikes, etc.)
- Support for multiple gesture datasets (DVS Gesture, SHREC, custom recordings)
- Visualisation of training curves and ABC convergence
- Model checkpointing and export to NIR (Neuromorphic Intermediate Representation)

---

## Project Structure

```
Gesture-recognition/
├── data/
│   ├── dataset_1/               # DVS128 Gesture Dataset
│   └── dataset_2/               
├── docs/
│   └── ai-audit/              
├── models/
│   ├── checkpoints/             # Saved model weights
│   └── exported_nir/            # Models exported to NIR format
├── src/
│   ├── core/                    # SNN model definition, training, evaluation
│   ├── optimization/            # ABC algorithm implementation
│   ├── pipeline/                
│   └── utils/                 
├── requirements.txt
└── README.md
```

---

## Installation & Setup


1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Gesture-recognition.git
   cd Gesture-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare datasets**  
   Place your datasets into `data/dataset_1/` and `data/dataset_2/`.  
   For public datasets (e.g., DVS Gesture), see **Datasets** section below.

4. **Run ABC optimisation**
   ```bash
   python src/pipeline/run_abc_optimization.py --dataset dataset_1 --epochs 50 --colony_size 20
   ```

5. **Train without optimisation** (using a predefined config)
   ```bash
   python src/core/train.py --config configs/default.yaml
   ```

6. **Evaluate the best model**
   ```bash
   python src/core/evaluate.py --checkpoint models/checkpoints/best_model.pth
   ```

---

## Datasets


Public **DVS128 Gesture Dataset** (recorded with an event camera, 11 gesture classes).  
  [IBM DVS Gesture](https://research.ibm.com/interactive/dvs-gesture-dataset)

Preprocessing scripts for converting event streams into SNN‑compatible formats are located in `src/utils/preprocess.py`.

---

## Hyperparameter Optimisation (ABC)

The **Artificial Bee Colony** algorithm mimics foraging behaviour of honey bees. In our pipeline:

### Tunable SNN Hyperparameters

- Learning rate
- Beta

The best hyperparameter configuration is saved to `models/exported_nir/abc_best_config.json`.

---

## Team

- *Saida Musaeva* – SNN architecture, ABC algorithm
- *Makhail Tereshchenko* – training pipeline, hyperparameter optimisation,
- *Anna Tihonova* – Data preparation, visualisation

