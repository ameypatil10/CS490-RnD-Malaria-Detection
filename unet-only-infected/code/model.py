import torch.nn as nn
import torch
import torch.nn.init as weight_init

class Conv(nn.Module):
  def __init__(self, in_ch, out_ch, padding):
    super(Conv, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(in_ch, out_ch, 3, padding=padding),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
      nn.Conv2d(out_ch, out_ch, 3, padding=padding),
      nn.BatchNorm2d(out_ch),
      nn.ReLU()
    )

    self._init_weights()
  
  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        weight_init.kaiming_normal_(m.weight, nonlinearity='relu')
  
  def forward(self, x):
    x = self.conv(x)
    return x

class down(nn.Module):
  def __init__(self, in_ch, out_ch, padding):
    super(down, self).__init__()
    
    self.down = nn.Sequential(
      nn.MaxPool2d(2),
      Conv(in_ch, out_ch, padding)
    )
  
  def forward(self, x):
    return self.down(x)

class up(nn.Module):
  def __init__(self, in_ch, out_ch, padding):
    super(up, self).__init__()

    self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, padding=0, stride=2)
    self.conv = Conv(in_ch, out_ch, padding=padding)
    
  def forward(self, upsample, downsample):
    upsample = self.up(upsample)
    
    diffY = downsample.size()[2] - upsample.size()[2]
    diffX = downsample.size()[3] - upsample.size()[3]
    upsample = nn.functional.pad(upsample, (diffX//2, diffX-diffX//2, \
      diffY//2, diffY-diffY//2))
    
    x = torch.cat([downsample, upsample], dim=1)
    y = self.conv(x)
    return self.conv(x)
    
class UNet(nn.Module):
  def __init__(self, in_channel=3, n_class=1):
    super(UNet, self).__init__()

    self.dconv1 = Conv(in_channel,64,padding=0)
    self.dconv2 = down(64,128,padding=0)
    self.dconv3 = down(128,256,padding=0)
    self.dconv4 = down(256,512,padding=0)
    
    self.bconv = down(512,1024,padding=0)

    self.uconv1 = up(1024,512,padding=0)
    self.uconv2 = up(512,256,padding=0)
    self.uconv3 = up(256,128,padding=0)
    self.uconv4 = up(128,64,padding=2)

    # 1x1 2D Convolution filter from 64 channels to n_channels
    self.reduce = nn.Sequential(\
      nn.Conv2d(64,n_class,1,padding=0),\
      nn.ReLU(),\
      nn.Conv2d(n_class,n_class,1,padding=0),\
      nn.Sigmoid()\
    )

    # self._init_weights()

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

  def forward(self, x):
    x1 = self.dconv1(x)
    x2 = self.dconv2(x1)
    x3 = self.dconv3(x2)
    x4 = self.dconv4(x3)
    x5 = self.bconv(x4)
    x6 = self.uconv1(x5,x4)
    x7 = self.uconv2(x6,x3)
    x8 = self.uconv3(x7,x2)
    x9 = self.uconv4(x8,x1)
    x10 = self.reduce(x9)
    return x10
