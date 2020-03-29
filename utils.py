import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pathlib
import math
import os


def check_file(path):
    # 检查文件是否存在
    if not os.path.exists(path):
        os.makedirs(path)
    return


def de_norm(norm_img):
    # 将图像从[-1, 1] --> [0, 255]
    img = (norm_img + 1) * 127.5
    img = np.array(img)
    img = img.astype(np.uint8)
    return img


def make_gird(inputs, denorm=False, padding=2, n_rows=6):
    # 将一个batch的图像合并为一整个大图
    if denorm:
        de_norm_img = de_norm(inputs)
    else:
        de_norm_img = inputs

    n_maps, x_maps, y_maps, c_maps = de_norm_img.shape  # [b, h, w, c]

    if c_maps == 1:
        de_norm_img = np.squeeze(de_norm_img, axis=-1)
    x_maps = min(n_rows, x_maps)  # 取最小值
    y_maps = int(math.ceil(float(n_maps) / x_maps))

    # 图像经过 padding 之后的高度和宽度
    height, width = int(de_norm_img.shape[1] + padding), int(de_norm_img.shape[2] + padding)

    # 初始化一整张大图
    grid = np.zeros(shape=(height * y_maps + padding, width * x_maps + padding, c_maps), dtype=np.uint8)
    k = 0
    for y in range(y_maps):
        for x in range(x_maps):
            if k >= n_maps:
                break
            grid[y * height + padding:(y + 1) * height,
                 x * width + padding:(x + 1) * width, :] = de_norm_img[k]
            k = k + 1
    return grid


def save_samples(gen_sample, save_path, epoch, step, i=0, color='rgb'):
    gen_sample = de_norm(gen_sample)
    img_name = 'epoch-{:06d}-step-{:06d}-num-{:02d}.jpg'.format(epoch, step, i)
    if color == 'rgb':
        plt.imsave(os.path.join(save_path, img_name), gen_sample)
    else:
        cv.imwrite(os.path.join(save_path, img_name), gen_sample)
    return


if __name__ == '__main__':
    import test_data

    cifar_data = test_data.get_data(batch_size=36, data_name='cifar10')
    for imgs, labels in cifar_data:
        de_norm_img_ = make_gird(imgs, denorm=True, padding=0)
        print(np.max(de_norm_img_))
        print(np.min(de_norm_img_))
        print(de_norm_img_.dtype)
        plt.imshow(de_norm_img_)
        plt.axis('off')
        plt.show()
        break
