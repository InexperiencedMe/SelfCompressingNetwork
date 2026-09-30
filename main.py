import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

EPOCHS = 200
BATCH_SIZE = 256
GAMMA = 0.05  # Larger = stronger pressure to use fewer bits

class QuantizedLinear(nn.Module): # Linear with no bias
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(outputs, inputs), a=5**0.5))
        self.bits   = nn.Parameter(torch.full((outputs, 1), 32.0))

        maxInt32 = 2**31 - 1 # 2 147 483 647
        initialStepSize = self.weight.detach().abs().amax(dim=1, keepdim=True) / maxInt32
        self.exponent = nn.Parameter(initialStepSize.clamp_min(1e-12).log2())

    def forward(self, x):
        stepSize = torch.exp2(self.exponent)
        bound = torch.exp2((self.bits.relu() - 1))

        slotValues = self.weight / stepSize
        clippedSlotValues = torch.clamp(slotValues, min = -bound, max = bound - 1)
        slotIntegers = clippedSlotValues + (clippedSlotValues.round() - clippedSlotValues).detach()  # Straight-Through Estimator
        return F.linear(x, slotIntegers * stepSize)

    def estimateBits(self):
        return self.weight.shape[1] * self.bits.relu().sum()


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
    print(f'Training on device: {device}')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train   = DataLoader(MNIST('data', train=True,  download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test    = DataLoader(MNIST('data', train=False, download=True, transform=transform), batch_size=BATCH_SIZE*2)

    with torch.device(device):
        layers = [QuantizedLinear(784, 256), QuantizedLinear(256, 128), QuantizedLinear(128, 10)]
        model = nn.Sequential(nn.Flatten(), layers[0], nn.ReLU(), layers[1], nn.ReLU(), layers[2])

    paramCount = sum(layer.weight.numel() for layer in layers)
    baselineBits = 32 * paramCount
    optimizer = torch.optim.AdamW([
        {'params': [layer.weight for layer in layers], 'lr': 0.001},
        {'params': [param for layer in layers for param in (layer.bits, layer.exponent)], 'lr': 0.02}])

    for epoch in range(EPOCHS + 1):
        if epoch > 0: # Epoch 0 evaluates the initial model before any training
            # Training
            model.train()
            for batch, (x, y) in enumerate(train, start=1):
                x, y = x.to(device), y.to(device)

                averageBitsCount = sum(layer.estimateBits() for layer in layers) / paramCount
                compressionsLoss = GAMMA * averageBitsCount

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
        print(f'Epoch {epoch:4d} | accuracy {accuracy:6.2f}% | estimated bits {bits:9_.0f} | {100 * bits / baselineBits:6.2f}% of 32-bit weights')
