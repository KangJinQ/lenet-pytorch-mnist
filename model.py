import torch
from torch import nn


class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        # 输入为[batch_size, 1, 28, 28]
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),  # ->[batch_size, 6, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # ->[batch_size,16,8,8]
            nn.ReLU(),
            nn.MaxPool2d(2),  # ->[batch_size,16,4,4]
        )
        self.flatten = nn.Flatten()
        self.Fc1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),  # full connect 16*4*4=256-->120
            nn.ReLU(),
        )
        self.Fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
        )
        self.Fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10),
            nn.ReLU(),
        )

    def forward(self, X):
        x = self.Conv1(X)
        x = self.Conv2(x)
        x = self.flatten(x)
        x = self.Fc1(x)
        x = self.Fc2(x)
        y = self.Fc3(x)
        return y


if __name__ == '__main__':
    test = torch.ones((3, 1, 28, 28))  # batch_size = 3
    model = MyLeNet()
    y = model(test)
    print(y)
