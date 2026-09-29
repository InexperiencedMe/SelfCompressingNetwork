import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

EPOCHS = 20
BATCH_SIZE = 256
GAMMA = 0.05  # Larger = stronger pressure to use fewer bits.

class QuantizedLinear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(outputs, inputs), a=5**0.5))
        self.bits   = nn.Parameter(torch.full((outputs, 1), 4.0))

        scale = self.weight.detach().abs().amax(dim=1, keepdim=True) / 7
        self.exponent = nn.Parameter(scale.clamp_min(1e-12).log2())

    def forward(self, x):
        # Each neuron shares one scale and bit budget across its weights.
        scale = torch.exp2(self.exponent)
        limit = torch.exp2((self.bits.relu() - 1))
        codes = torch.clamp(self.weight / scale, min=-limit, max=limit - 1)
        codes = codes + (codes.round() - codes).detach()  # Straight-through gradient.
        return F.linear(x, codes * scale)

    def estimateBits(self):
        return self.weight.shape[1] * self.bits.relu().sum()


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train   = DataLoader(MNIST('data', train=True,  download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test    = DataLoader(MNIST('data', train=False, download=True, transform=transform), batch_size=BATCH_SIZE*2)

    with torch.device(device):
        layers = [QuantizedLinear(784, 256), QuantizedLinear(256, 128), QuantizedLinear(128, 10)]
        model = nn.Sequential(nn.Flatten(), layers[0], nn.ReLU(), layers[1], nn.ReLU(), layers[2])

    paramCount = sum(layer.weight.numel() for layer in layers)
    optimizer = torch.optim.AdamW([
        {'params': [layer.weight for layer in layers], 'lr': 0.001},
        {'params': [param for layer in layers for param in (layer.bits, layer.exponent)], 'lr': 0.02}])

    accuracies, bitCounts = [], []
    print(f'Training on device: {device}')
    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        for batch, (x, y) in enumerate(train, start=1):
            x, y = x.to(device), y.to(device)
            
            progress = epoch - 1 + batch / len(train) # First epoch: learn digits. Next two: ramp compression pressure
            gamma = GAMMA * min(1.0, max(0.0, (progress - 1) / 2))
            averageBitsCount = sum(layer.estimateBits() for layer in layers) / paramCount
            compressionsLoss = gamma * averageBitsCount

            classificationLoss = F.cross_entropy(model(x), y)
            loss = classificationLoss + compressionsLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.inference_mode():
            correct = 0
            for x, y in test:
                prediction = model(x.to(device)).argmax(dim=1)
                correct += (prediction == y.to(device)).sum().item()
            accuracy = 100 * correct / len(test.dataset)
            bits = sum(layer.estimateBits() for layer in layers).item()
        accuracies.append(accuracy)
        bitCounts.append(bits)
        print(f'Epoch {epoch:2d} | accuracy {accuracy:.2f}% | estimated bits {bits:,.0f}')

    # Summary plot
    epochs = range(1, EPOCHS + 1)
    fig, (ax_accuracy, ax_bits) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), layout='constrained')
    ax_accuracy.plot(epochs, accuracies, 'o-', color='tab:blue', markersize=3)
    ax_accuracy.set(ylabel='Test accuracy (%)', ylim=(95, 100), title='MNIST self-compression')
    ax_bits.plot(epochs, bitCounts, 'o-', color='tab:red', markersize=3)
    ax_bits.set(xlabel='Epoch', ylabel='Estimated weight bits (bit)', ylim=(0, None))
    for ax in (ax_accuracy, ax_bits):
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
    plt.show()
