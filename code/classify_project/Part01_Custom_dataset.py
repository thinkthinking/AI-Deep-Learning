import glob
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import T_co
from torchvision import datasets, transforms
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch


def get_image_pd(img_root,defect_label):
    # 利用glob指令获取图片列表（/*的个数根据文件构成确定）
    img_list = glob.glob(img_root + "/*/*.jpg")
    print('-----img_list-------', len(img_list))
    # 利用DataFrame指令构建图片列表的字典，即图片列表的序号与其路径一一对应
    image_pd = pd.DataFrame(img_list, columns=["ImageName"])
    # 获取文件夹名称，也可以认为是标签名称
    image_pd["label_name"] = image_pd["ImageName"].apply(lambda x: x.split("/")[-2])
    # 将标签名称转化为数字标记
    image_pd["label"] = image_pd["label_name"].apply(lambda x: defect_label[x])
    print(image_pd["label"].value_counts())
    return image_pd


class Mydataset(data.Dataset):
    def __init__(self, anno_pd, transform=None):
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.transforms = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.paths[index]
        image = Image.open(img_path).convert('RGB')
        image_transform = self.transforms(image)
        label = int(self.labels[index])
        return image_transform, label


def show_batch_images(sample_batch):
    labels_batch = sample_batch[1]
    images_batch = sample_batch[0]

    for i in range(6):
        label_ = labels_batch[i].item()
        # [3,224,224]  --->  transpose --> (224,224,3)
        image_ = np.transpose(images_batch[i], (1, 2, 0))
        ax = plt.subplot(1, 6, i + 1)
        ax.imshow(image_)
        ax.set_title(str(label_))
        ax.axis('off')


if __name__ == '__main__':
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

    img_root_train = './data/NEU-CLS-64'
    all_pd = get_image_pd(img_root_train,defect_label)
    train_pd, val_pd = train_test_split(all_pd, test_size=0.2, random_state=54, stratify=all_pd["label"])

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])

    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
    # train_pd.to_csv('./train.csv')
    # val_pd.to_csv('./val.csv')

    train_datsset = Mydataset(train_pd, transform=train_transform)
    val_dataset = Mydataset(val_pd, transform=val_transform)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_datsset, batch_size=6, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=6, shuffle=False)

    plt.figure()
    for i_batch, sample_batch in enumerate(val_dataloader):
        show_batch_images(sample_batch)
        plt.show()
        break