import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):

        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            print("---------img------", img.size)
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)

            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    plt.rcParams["savefig.bbox"] = 'tight'
    orig_img = Image.open(Path('assets') / 'cat_6.jpg')
    # orig_img = Image.open(Path('assets') / 'astronaut.jpg')
    torch.manual_seed(100)

    # 1、pad 填充
    padding_imgs = [transforms.Pad(padding,fill= 0,padding_mode='constant')(orig_img) for padding in (10, 30, 50, 100)]
    plot(padding_imgs)


    # 2、重置大小
    # resized_imgs = [transforms.Resize(size=(size,size))(orig_img) for size in (30, 50, 80, orig_img.size[0])]
    # plot(resized_imgs)


    # 3、中心裁剪
    # center_crops = [transforms.CenterCrop(size=size)(orig_img) for size in (200, 300, 400, orig_img.size[0])]
    # plot(center_crops)


    # 4、在原图片的四个角和中心各截取一幅大小为 size 的图片
    # (top_left, top_right, bottom_left, bottom_right, center) = transforms.FiveCrop(size=(300, 300))(orig_img)
    # plot([top_left, top_right, bottom_left, bottom_right, center])


    # 5、灰度图
    # gray_img = transforms.Grayscale()(orig_img)
    # plot([gray_img], cmap='gray')


    # 6、随机旋转
    # rotater = transforms.RandomRotation(degrees=(0, 180))
    # rotated_imgs = [rotater(orig_img) for _ in range(4)]
    # plot(rotated_imgs)


    # 7、随机裁剪
    # cropper = transforms.RandomCrop(size=(400,400),pad_if_needed=True)
    # crops = [cropper(orig_img) for _ in range(4)]
    # plot(crops)


    # 8、将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为指定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
    # resize_cropper = transforms.RandomResizedCrop(size=(320, 320),scale=(0.5,0.8),ratio=(0.5,0.5))
    # resized_crops = [resize_cropper(orig_img) for _ in range(4)]
    # plot(resized_crops)


    # 9、随机水平翻转
    # rotater = transforms.RandomHorizontalFlip()
    # rotated_imgs = [rotater(orig_img) for _ in range(4)]
    # plot(rotated_imgs)


    # 10、随机垂直翻转
    # rotater = transforms.RandomVerticalFlip()
    # rotated_imgs = [rotater(orig_img) for _ in range(4)]
    # plot(rotated_imgs)


    # 11、改变图像的属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
    # rotater = transforms.ColorJitter(brightness=0,contrast=0,saturation=1.7)
    # rotated_imgs = [rotater(orig_img) for _ in range(4)]
    # plot(rotated_imgs)

