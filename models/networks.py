import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))

class StegoEncoder(nn.Module):
    def __init__(self):
        super(StegoEncoder, self).__init__()
        # ... (Keep your init code exactly the same as before) ...
        # Input: 4 channels -> Output: 3 channels
        self.enc1 = nn.Conv2d(4, 64, 4, 2, 1)  
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1) 
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1) 
        
        self.res_block = ResBlock(256)
        
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) 
        self.dec2 = nn.ConvTranspose2d(256, 64, 4, 2, 1)  
        self.dec3 = nn.ConvTranspose2d(128, 32, 4, 2, 1)  

        self.final = nn.Conv2d(32, 3, 3, 1, 1)
        self.tanh = nn.Tanh() # Squashes output to -1...1
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1) 
        
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        
        b = self.res_block(e3)
        
        d1 = self.relu(self.dec1(b))
        d1 = torch.cat([d1, e2], dim=1) 
        d2 = self.relu(self.dec2(d1))
        d2 = torch.cat([d2, e1], dim=1) 
        d3 = self.relu(self.dec3(d2))
        
        # --- THE FIX IS HERE ---
        # 1. Tanh forces the noise to be between -1 and 1
        delta = self.tanh(self.final(d3))
        
        # 2. Multiply by 0.03.
        # This means the AI can only change a pixel by 3% max.
        # Purple Haze is IMPOSSIBLE with this line.
        stego = torch.clamp(cover + (delta * 0.03), 0, 1)
        
        return stego

class StegoDecoder(nn.Module):
    def __init__(self):
        super(StegoDecoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)
        self.final = nn.Conv2d(64, 1, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, stego):
        x = self.relu(self.conv1(stego))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.final(x)
        return x
