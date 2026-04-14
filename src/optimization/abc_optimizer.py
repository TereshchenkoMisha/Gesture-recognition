import torch
import torch.nn as nn
import numpy as np
import sys
import os
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.model.model import GestureSNN, calc_accuracy, device
from src.pipeline.dataloader.emg_pkl import get_emg_pkl_loaders

print("Loading datasets for the swarm...")
train_loader, test_loader = get_emg_pkl_loaders(
    pkl_path="data/EMG_DVS/relax21_cropped_dvs_emg_spikes.pkl",
    batch_size=8,
    train_subjects=list(range(1, 17)),
    test_subjects=list(range(17, 22)),
    num_frames=20
)
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

base_model = GestureSNN(input_channels=2, num_classes=5)
INITIAL_WEIGHTS = copy.deepcopy(base_model.state_dict())

def fitness_function(params, epochs=1):
    beta_val, lr_val = params
    
    local_model = GestureSNN(input_channels=2, num_classes=5).to(device)
    local_model.load_state_dict(INITIAL_WEIGHTS)

    for module in local_model.modules():
        if hasattr(module, 'beta'):
            module.beta = torch.tensor(beta_val, dtype=torch.float32, device=device)

    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=float(lr_val), weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    local_model.train()
    
    for epoch in range(epochs):
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            data = data.transpose(0, 1)
            
            local_optimizer.zero_grad()
            spk_out = local_model(data)
            spk_count = spk_out.sum(dim=0)
            
            loss = loss_fn(spk_count, targets)
            loss.backward()
            local_optimizer.step()

    acc = calc_accuracy(local_model, test_loader)
    error = 1.0 - acc 
    
    print(f"Evaluated [beta={beta_val:.3f}, lr={lr_val:.5f}] -> Accuracy: {acc*100:.2f}%")
    return error


class SimpleABC:
    def __init__(self, num_bees, bounds, iter_max, limit):
        self.num_food = num_bees // 2 
        self.bounds = np.array(bounds)
        self.iter_max = iter_max
        self.limit = limit 
        self.dim = len(bounds)
        
        self.foods = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_food, self.dim))
        self.fitness = np.zeros(self.num_food)
        self.trials = np.zeros(self.num_food) 
        
        self.best_food = np.zeros(self.dim)
        self.best_score = float('inf')

    def optimize(self, fitness_func):
        print("Initializing scout bees...")
        for i in range(self.num_food):
            self.fitness[i] = fitness_func(self.foods[i])
            if self.fitness[i] < self.best_score:
                self.best_score = self.fitness[i]
                self.best_food = np.copy(self.foods[i])

        for it in range(self.iter_max):
            print(f"\n--- ABC Iteration {it+1}/{self.iter_max} ---")
            
            for i in range(self.num_food):
                partner = np.random.choice([p for p in range(self.num_food) if p != i])
                phi = np.random.uniform(-1, 1, self.dim)
                
                new_food = self.foods[i] + phi * (self.foods[i] - self.foods[partner])
                new_food = np.clip(new_food, self.bounds[:, 0], self.bounds[:, 1])
                
                new_score = fitness_func(new_food)
                if new_score < self.fitness[i]:
                    self.foods[i] = new_food
                    self.fitness[i] = new_score
                    self.trials[i] = 0 
                else:
                    self.trials[i] += 1

            fit_probs = 1.0 / (1.0 + self.fitness) 
            probs = fit_probs / np.sum(fit_probs)

            t = 0
            i = 0
            while t < self.num_food:
                if np.random.rand() < probs[i]: 
                    t += 1
                    partner = np.random.choice([p for p in range(self.num_food) if p != i])
                    phi = np.random.uniform(-1, 1, self.dim)
                    
                    new_food = self.foods[i] + phi * (self.foods[i] - self.foods[partner])
                    new_food = np.clip(new_food, self.bounds[:, 0], self.bounds[:, 1])
                    
                    new_score = fitness_func(new_food)
                    if new_score < self.fitness[i]:
                        self.foods[i] = new_food
                        self.fitness[i] = new_score
                        self.trials[i] = 0
                    else:
                        self.trials[i] += 1
                i = (i + 1) % self.num_food

            for i in range(self.num_food):
                if self.trials[i] > self.limit:
                    print(f"[!] Source {i} exhausted. Scout bee deployed!")
                    self.foods[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                    self.fitness[i] = fitness_func(self.foods[i])
                    self.trials[i] = 0

            min_idx = np.argmin(self.fitness)
            if self.fitness[min_idx] < self.best_score:
                self.best_score = self.fitness[min_idx]
                self.best_food = np.copy(self.foods[min_idx])
                
            print(f">>> Best this iteration: Accuracy {(1.0 - self.best_score)*100:.2f}% | [beta={self.best_food[0]:.3f}, lr={self.best_food[1]:.5f}]")

        return self.best_food

if __name__ == "__main__":
    search_bounds = [(0.5, 0.99), (0.0001, 0.002)]
    
    abc = SimpleABC(num_bees=6, bounds=search_bounds, iter_max=3, limit=2)
    best_params = abc.optimize(fitness_function)
    
    print("\npriveet!")
    print(f"The best parameters -> beta: {best_params[0]:.4f}, lr: {best_params[1]:.5f}")
