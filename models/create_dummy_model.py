import torch
import torch.nn as nn

# Define same model structure as app.py
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Instantiate model
model = SimpleCNN()

# Save it to your models/ folder
torch.save(model.state_dict(), "models/cnn_model.pth")

print("Dummy CNN model saved.")
