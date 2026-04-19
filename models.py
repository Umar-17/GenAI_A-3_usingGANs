import torch
import torch.nn as nn

class DCGAN_Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(DCGAN_Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.net(input)

class WGAN_Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(WGAN_Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.net(input)

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x, skip):
        x = self.conv(x)
        return torch.cat([x, skip], dim=1)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        self.e0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2)
        )
        self.e1 = DownBlock(64, 128)
        self.e2 = DownBlock(128, 256)
        self.e3 = DownBlock(256, 512)
        self.e4 = DownBlock(512, 512)
        self.e5 = DownBlock(512, 512)
        self.e6 = DownBlock(512, 512)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=True),
            nn.ReLU()
        )
        
        self.d0 = UpBlock(512, 512, dropout=True)
        self.d1 = UpBlock(1024, 512, dropout=True)
        self.d2 = UpBlock(1024, 512, dropout=True)
        self.d3 = UpBlock(1024, 512, dropout=False)
        self.d4 = UpBlock(1024, 256, dropout=False)
        self.d5 = UpBlock(512, 128, dropout=False)
        self.d6 = UpBlock(256, 64, dropout=False)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        
        b = self.bottleneck(e6)
        
        d0 = self.d0(b, e6)
        d1 = self.d1(d0, e5)
        d2 = self.d2(d1, e4)
        d3 = self.d3(d2, e3)
        d4 = self.d4(d3, e2)
        d5 = self.d5(d4, e1)
        d6 = self.d6(d5, e0)
        
        return self.final(d6)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)