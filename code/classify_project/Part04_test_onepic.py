import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import resnet50
import numpy as np



def predict(model, test_image_name):
    transform = image_transforms['test']
    test_image = Image.open(test_image_name).convert('RGB')
    print("--------test_image--------shape", test_image.size)
    img_pil_1 = np.array(test_image)  # (H x W x C), [0, 255], RGB
    print(img_pil_1.shape)

    draw = ImageDraw.Draw(test_image)

    test_image_tensor = transform(test_image)
    print("--------test_image_tensor--------shape",test_image_tensor.shape)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 64, 64).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 64, 64)

    with torch.no_grad():
        model.eval()

        out = model(test_image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(1, dim=1)
        print("-----------topclass-------",topclass)
        print("------------topk------------",topk)
        text =" " + str(topk.cpu().numpy()[0][0])
        font = ImageFont.truetype('./simsun.ttc', 36)
        draw.text((0, 0), text, (255, 0, 0), font=font)
        test_image.show()


if __name__ == '__main__':
    image_transforms = {
        'test': transforms.Compose([
            transforms.Resize(size=64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }


    # 1、 如果保存整个模型是(all_ckpt_0.pth)
    # torch.save(model, './model_parameter/test/ckpt_%s.pth' % (str(epoch)))
    # 加载模型应该使用如下的方法进行预测
    model = torch.load('./model_parameter/test/all_ckpt_0.pth',map_location=torch.device('cpu'))
    predict(model, './data/NEU-CLS-64/sp/1.jpg')


    # 2、 如果只保存模型参数(ckpt_0.pth)
    # state_dict = model.state_dict()
    # torch.save(state_dict, './model_parameter/test/ckpt_%s.pth' % (str(epoch)))
    # 加载模型应该使用如下的方法进行预测
    # model = resnet50(pretrained=True)
    # model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    # model.fc = torch.nn.Linear(model.fc.in_features, 9)
    # state_dict = torch.load('./model_parameter/test/ckpt_0.pth',map_location=torch.device('cpu'))
    # layer4.2.bn3.weight,在cpu上训练的没有module
    # # 加载参数
    # model.load_state_dict(state_dict)
    # predict(model, './data/NEU-CLS-64/sp/1.jpg')


    # 3、 模型在gpu上训练并只保存网络参数state_dict
    # model = resnet50(pretrained=True)
    # model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    # model.fc = torch.nn.Linear(model.fc.in_features, 2)
    #
    # state_dict = torch.load('./model_parameter/test/checkpoint.pth',map_location=torch.device('cpu'))
    # # 'module.layer4.2.bn3.weight
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # 去掉 `module.`
    #     new_state_dict[name] = v
    # # 加载参数
    # model.load_state_dict(new_state_dict)
    # predict(model, './data/NEU-CLS-64/gg/5.jpg')