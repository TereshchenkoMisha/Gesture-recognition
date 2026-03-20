#pip install snntorch
#pip install tonic
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta = 0.95
spike_grad = surrogate.fast_sigmoid() # Surrogate gradient for backpropagation

train_loader = []
test_loader = []


class GestureSNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [Batch, 2 channels (on/off), 128, 128]
        # Layer 1: 4x4 downsampling for DVS data
        self.conv1 = nn.Conv2d(2, 16, kernel_size=4, stride=4)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

        #Layer 2: extract features
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)

        # nn.Flattern() smoothing converts a multidimensional tensor into a one-dimensional one
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 8 * 8, 11)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):

     # Initialize membrane potentials for LIF neurons
     #membrane potential define when neuron will work
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = [] # Record output spikes over time


        # Iterate through time steps: x shape [Time, Batch, Channels, H, W]
        for step in range(x.size(0)):
            current_x = x[step]


            cur1 = self.pool1(self.conv1(current_x))
            spk1, mem1 = self.lif1(cur1, mem1)


            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)


            cur3 = self.fc1(self.flatten(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)

        return torch.stack(spk3_rec, dim=0)
#Here is the model,optimizer and loss function
model = GestureSNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def calc_accuracy(net, dataloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)

            data = data.transpose(0, 1)

            spk_out = net(data)

            # Prediction based on the highest spike count over T steps
            _, predicted = spk_out.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total if total > 0 else 0

# Train Loop ;)
epochs = 10


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i, (data, targets) in enumerate(train_loader):
        # tranport
        data, targets = data.to(device), targets.to(device)

        # [Batch, Time, Channels, H, W].
        data = data.transpose(0, 1)
        #optimizer.zero_grad() resets gradients from previous steps 
        optimizer.zero_grad()

        #get model prediction
        spk_out = model(data)
        spk_count = spk_out.sum(dim=0)

        #calculate loss function
        loss = loss_fn(spk_count, targets)

        #back propogation
        loss.backward()

        #step of gradient descent 
        optimizer.step()

        total_loss += loss.item()
        

        if i % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")


    val_acc = calc_accuracy(model, test_loader)

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    print(f"The end of the epoch {epoch+1} | Average Loss: {avg_loss:.4f} | Accurancy: {val_acc*100:.2f}% ---")

