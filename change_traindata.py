from torchvision import datasets
from torchvision.transforms import functional as F
import random
import matplotlib.pyplot as plt

# Mnistデータの取得
mnist = datasets.MNIST(root='mnist_data', train=True, download=True)

# 画像を回転や平行移動させる
def augment_image(img):
    angle = random.uniform(-10, 10)  # 回転角度
    translate = (random.randint(-2, 2), random.randint(-2, 2))  # x, y方向に±2ピクセル

    # 拡張（ランダム回転＋平行移動）
    img = F.affine(img, angle=angle, translate=translate, scale=1.0, shear=0)
    return img
