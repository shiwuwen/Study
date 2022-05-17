import  torch
from    torch import nn

class LeNet5(nn.Module):

    def __init__(self) -> None:
        super(LeNet5, self).__init__()
        
        # 卷积部分
        self.conv_unit = nn.Sequential(
            # x:[b, 3, 32, 32] => [b, 6, 28, 28]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # [b, 6, 28, 28] => [b, 6, 14, 14]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [b, 6, 14, 14] => [b, 16, 10, 10]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # [b, 16, 10, 10] => [b, 16, 5, 5]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # 全连接部分
        self.fc_unit = nn.Sequential(
            # [b, 16*5*5] => [b, 120]
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            # [b, 120] => [b, 84]
            nn.Linear(120, 84),
            nn.ReLU(),
            # [b, 84, 10]
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        # 前向传播
        x = self.conv_unit(x)
        x = x.view(-1, 16*5*5)
        logits = self.fc_unit(x)

        return logits

def main():
    net = LeNet5()
    x = torch.randn(2, 3, 32, 32)
    out = net.forward(x)

    print(out.size())


if __name__ == '__main__':
    main()