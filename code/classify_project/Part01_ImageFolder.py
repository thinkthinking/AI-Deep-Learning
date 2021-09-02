import torch
from torchvision import  datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

def show_batch_images(sample_batch):
    labels_batch = sample_batch[1]
    images_batch = sample_batch[0]

    for i in range(6):
        label_ = labels_batch[i].item()
        image_ = np.transpose(images_batch[i], (1, 2, 0))
        ax = plt.subplot(1, 6, i + 1)
        ax.imshow(image_)
        ax.set_title(str(label_))
        ax.axis('off')


if __name__ == '__main__':

    # 定义训练集transform
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=80),
        transforms.ToTensor(),

    ])

    # 定义验证集transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    batch_size = 6

    # 训练集路径
    train_dir = 'flowers/train/'
    # 进行数据增强后的训练集
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transform)
    # 训练集加载器，每次加载batch_size个样本数据
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=False)

    # 验证集路径
    val_dir = './flowers/val'
    # 进行数据增强后的验证集
    val_datasets = datasets.ImageFolder(val_dir, transform=val_transform)
    # 验证集加载器，每次加载batch_size个样本数据
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)

    # 展示
    for i_batch, sample_batch in enumerate(train_dataloader):
        show_batch_images(sample_batch)
        plt.show()
        break