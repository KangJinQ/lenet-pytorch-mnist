import torch.cuda
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageOps
import numpy as np
import os

images_path = 'images'
model_dir = 'models'
model_version = 'mnist_0.98.pkl'


def GetImages(path):
    filename_list = os.listdir(path)
    return filename_list


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(model_dir, model_version)
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    # 加载图片
    images_list = GetImages(images_path)
    plt.subplots(2, 5, figsize=(10, 5))
    target_list = []
    num = 1  # 用于计数图片显示位置
    for img_name in images_list:
        target_list.append(img_name.split('.')[0])  # 将标签加入标签列表
        # 单通道读入图片
        img = Image.open(os.path.join(images_path, img_name)).convert('L')
        img = img.resize((28, 28))
        # 颜色反转
        img_inv = ImageOps.invert(img)
        # img_binary = img.convert('1')  # 二值化颜色
        img_array = np.array(img_inv).astype(np.float32)
        img = np.reshape(img_array, (1, 1, 28, 28))
        # img = np.expend_dims(img_array, 0)  # dim->[1, 28, 28]
        # img = np.expend_dims(img, 0)  # dim->[1, 1, 28, 28]
        img = torch.from_numpy(img)  # -->torch
        img = img.to(device)
        output = model(img)
        output = output.to('cpu')
        output = output.detach().numpy()
        predict_y = np.argmax(output, axis=-1)

        plt.subplot(2, 5, num)
        num += 1
        # img_show = img_inv.convert('1')  # bgr->rgb
        img_show = img_inv.convert('RGBA')
        plt.imshow(img_show)
        plt.title('predict:' + str(predict_y))
        plt.axis('off')
    plt.show()
