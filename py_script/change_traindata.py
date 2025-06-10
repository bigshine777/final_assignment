import struct
import numpy as np
from torchvision import datasets
from torchvision.transforms import functional as F
import random

mnist = datasets.MNIST(root="mnist_data", train=True, download=True)

num_images = 60000
height, width = 28, 28

images = []


# 画像データを並行移動させたり傾けさせたりする関数
def change_images(img):
    angle = random.uniform(-10, 10)
    translate = (random.randint(-2, 2), random.randint(-2, 2))
    img = F.affine(img, angle=angle, translate=translate, scale=1.0, shear=0)
    return img


# 元の画像の全てを変換
for i in range(num_images):
    img, label = mnist[i % 60000]
    img = change_images(img)
    img_np = np.array(img, dtype=np.uint8)
    images.append(img_np)

# 画像を1つのバイナリとしてまとめる
image_data = np.stack(images)
image_data = image_data.reshape(num_images, -1)

# ヘッダー情報を書く（magic number: 2051）
with open("binary/increased-images-idx3-ubyte", "wb") as f:
    f.write(struct.pack(">IIII", 2051, num_images, width, height))

    for i in range(num_images):
        f.write(image_data[i].tobytes())
