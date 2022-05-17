import  torch
from    torch import nn
import  torch.nn.functional as F

class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out) -> None:
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)

        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(ch_out)
            )
        

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.extra(x) + out

        return out


class ResNet(nn.Module):

    def __init__(self) -> None:
        super(ResNet, self).__init__()

        self.conv = nn.Sequential(
            # [b, 3, 32, 32] => [b, 64, 32, 32]
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),

            # [b, 64, 32, 32] => [b, 128, 16, 16]
            ResBlk(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [b, 128, 16, 16] => [b, 256, 8, 8]
            ResBlk(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [b, 256, 8, 8] => [b, 512, 4, 4]
            ResBlk(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [b, 512, 4, 4] => [b, 512, 2, 2]
            ResBlk(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(
            # [b, 512*2*2] => [b, 512]
            nn.Linear(256*2*2, 512),
            nn.ReLU(),
            # [b, 512] => [b, 128]
            # nn.Linear(512, 128),
            # nn.ReLU(),
            # [b, 128] => [b, 10]
            nn.Linear(512, 10)
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)

        return out


def main():
    resnet = ResNet()
    x = torch.randn(2, 3, 32, 32)
    out = resnet(x)
    print(out.shape)


if __name__ == "__main__":
    main()

