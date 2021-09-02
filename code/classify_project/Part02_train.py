import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from Part01_Custom_dataset import get_image_pd
from torchvision.models import resnet50, resnet101, vgg16
from Part01_Custom_dataset import Mydataset
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import os
from Part03_Focal_loss import FocalLoss


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


def draw_loss_and_accuracy(Loss_list, Accuracy_list,name):
    plt.figure()
    x1 = range(0, len(Loss_list))
    x2 = range(0, len(Accuracy_list))
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. step')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. step')
    plt.ylabel('Test loss')
    plt.savefig(f'./{name}.jpg')



if __name__ == '__main__':
    img_root_train = './data/NEU-CLS-64'
    # 与数字标记一一对应
    defect_label = {
        'cr': '0',
        'gg': '1',
        'in': '2',
        'pa': '3',
        'ps': '4',
        'rp': '5',
        'rs': '6',
        'sc': '7',
        'sp': '8'
    }
    all_pd = get_image_pd(img_root_train,defect_label)
    train_pd, val_pd = train_test_split(all_pd, test_size=0.2, random_state=53, stratify=all_pd["label"])

    train_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([transforms.Resize(64),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
    # train_pd.to_csv('./train.csv')
    # val_pd.to_csv('./val.csv')

    # 训练集加载器
    train_datsset = Mydataset(train_pd, transform=train_transform)
    val_dataset = Mydataset(val_pd, transform=val_transform)
    # 验证集加载器
    train_dataloader = torch.utils.data.DataLoader(dataset=train_datsset, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    num_epoch = 1
    device = 'cpu'
    # # 网络模型resnet
    # model = ResnetModel(n_class=9)
    # model.to(device=device)
    # optim_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    model = resnet50(pretrained=True)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    model.fc = torch.nn.Linear(model.fc.in_features, 9)
    model.to(device=device)
    optim_sgd = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    acc_all = []
    loss_all = []
    val_acc = []
    val_loss = []

    for epoch in range(num_epoch):
        print("-----------epoch-------", epoch)
        model.train()
        for i, data in enumerate(train_dataloader):
            img, label = data
            img = img.float().to(device)
            label = label.long().to(device)
            # 前向传播
            output = model(img)
            # 计算损失
            loss = criterion(output, label)
            # 梯度清零
            optim_sgd.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optim_sgd.step()


            _, predicted = torch.max(output.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            total = label.size(0)
            correct = (predicted == label).sum().item()
            acc = correct / total
            print("-------pred----------", predicted)
            print("---------PRED_COUNT---------", correct)
            acc_all.append(acc)
            loss_all.append(loss)


        if epoch % 2 == 0:
            with torch.no_grad():
                model.eval()  # 使用验证集时关闭梯度计算
                for data in val_dataloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    acc = correct/total
                    val_acc.append(acc)
                    val_loss.append(loss)


    # loss-acc 可视化
    draw_loss_and_accuracy(loss_all, acc_all,'train_acc_loss')
    draw_loss_and_accuracy(val_loss, val_acc,'val_acc_loss')

