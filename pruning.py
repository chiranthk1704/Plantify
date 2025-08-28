import torch
import torch.nn as nn
import torch.quantization
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

def prune_model(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            with torch.no_grad():
                weight_copy = module.weight.data.abs().clone()
                threshold = weight_copy.view(-1).topk(int(0.1 * weight_copy.numel())).values[-1]
                mask = weight_copy > threshold
                module.weight.data.mul_(mask)
            print(f"Pruned {name} - {module}")
    return model

def quantize_model(model):
    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    input_tensor = torch.randn(1, 3, 64, 64)  
    model(input_tensor)

    torch.quantization.convert(model, inplace=True)
    return model

leaf_model = torch.load('plant_model.pth', map_location=torch.device('cpu'))

leaf_model = prune_model(leaf_model)

quantized_model = quantize_model(leaf_model)

torch.save(quantized_model.state_dict(), 'reduced_plant_model.pth')

print("Model has been pruned, quantized, and saved successfully!")
