import torch.nn as nn
import torch

# 定义虚线的的Connection部分，表示通道不同
class ConvBlock(nn.Module):
    def __init__(self, in_channels, filter, s, f):
        super(ConvBlock, self).__init__()
        # 卷积->BN->RELU ->卷积->BN->RELU->卷积->BN
        F1, F2, F3 = filter
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=F1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1), nn.ReLU(True),
            nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=f, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F2), nn.ReLU(True),
            nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(F3))
        self.shortcut_1 = nn.Conv2d(in_channels=in_channels, out_channels=F3, kernel_size=1, stride=s, padding=0,
                                    bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        shortcut_1 = self.shortcut_1(X)
        batch_1 = self.batch_1(shortcut_1)
        X = self.stage(X)
        X = X + batch_1
        X = self.relu_1(X)
        return X


class IdentifyBlock(nn.Module):
    def __init__(self, in_channel, filters):
        super(IdentifyBlock, self).__init__()
        F1, F2, F3 = filters
        # 1*1 conv -> bn -> relu ->3*3 conv ->bn ->relu ->1*1 conv->bn
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=F1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1), nn.ReLU(True),
            nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F2), nn.ReLU(True),
            nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3))
        self.relu = nn.ReLU(True)

    def forward(self, X):
        shortcut_1 = X
        X = self.stage(X)
        X = shortcut_1 + X
        X = self.relu(X)
        return X


class ResnetModel(nn.Module):
    def __init__(self,n_classes):
        super(ResnetModel, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.stage2 = nn.Sequential(ConvBlock(in_channels=64,filter=[64,64,256],s=1),
                                    IdentifyBlock(in_channel=256,filters=[64,64,256]),
                                    IdentifyBlock(in_channel=256,filters=[64,64,256]))
        self.stage3 = nn.Sequential(ConvBlock(in_channels=256,filter=[128,128,512],s=2),
                                    IdentifyBlock(in_channel=512,filters=[128,128,512]),
                                    IdentifyBlock(in_channel=512,filters=[128,128,512]),
                                    IdentifyBlock(in_channel=512,filters=[128,128,512]),
                                    )
        self.stage4 = nn.Sequential(ConvBlock(in_channels=512,filter=[256,256,1024],s=2),
                                    IdentifyBlock(in_channel=1024,filters=[256,256,1024]),
                                    IdentifyBlock(in_channel=1024,filters=[256,256,1024]),
                                    IdentifyBlock(in_channel=1024,filters=[256,256,1024]),
                                    IdentifyBlock(in_channel=1024,filters=[256,256,1024]),
                                    IdentifyBlock(in_channel=1024,filters=[256,256,1024]),
                                    )
        self.stage5 = nn.Sequential(ConvBlock(in_channels=1024,filter=[512,512,2048],s=2),
                                    IdentifyBlock(in_channel=2048,filters=[512,512,2048]),
                                    IdentifyBlock(in_channel=2048,filters=[512,512,2048]),
                                    IdentifyBlock(in_channel=2048,filters=[512,512,2048]))

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=2048,out_features=n_classes)

    def forward(self, X):
        output = self.stage2(X)
        output = self.stage3(output)
        output = self.stage4(output)
        output = self.stage5(output)
        output = self.pool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output
