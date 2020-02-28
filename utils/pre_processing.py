#=========================================================#
#                                                         #
#        预 处 理 原 始 眼 底 视 网 膜 图 像                #
#                                                         #
#=========================================================#

import cv2
import numpy as np
from PIL import Image


#===== 将图像转为灰度图 =====#
def rgb2gray(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[3] == 3)  # check the channel is 3
    bn_imgs = imgs[:, :, :, 0] * 0.299 + imgs[:, :, :, 1] * 0.587 + imgs[:, :, :, 2] * 0.114
    bn_imgs = np.reshape(bn_imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))

    return bn_imgs


#===== 提取绿色通道 =====#
def extract_green_channel(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[3] == 3)  # check the channel is 3
    bn_imgs = imgs[:, :, :, 1]
    bn_imgs = np.reshape(bn_imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))

    return bn_imgs


#===== 直方图均衡化 =====#
def histo_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[3] == 1)  # check the channel is 1
    imgs_equalized = np.empty(imgs.shape)

    for i in range(imgs.shape[0]):
        imgs_equalized[i, :, :, 0] = cv2.equalizeHist(np.array(imgs[i, :, :, 0], dtype=np.uint8))

    return imgs_equalized


#===== 对比度受限的自适应直方图均衡化 =====#
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[3] == 1)  # check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)

    for i in range(imgs.shape[0]):
        imgs_equalized[i, :, :, 0] = clahe.apply(np.array(imgs[i, :, :, 0], dtype=np.uint8))

    return imgs_equalized


#===== 数据集标准化 =====#
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[3] == 1)  # check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std  # 标准化
    # 归一化
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))*255

    return imgs_normalized


#====== 伽马变化 ======#
def adjust_gamma(imgs, gamma = 1.0):
    assert (len(imgs.shape) == 4)  #4D arrays
    assert (imgs.shape[3] == 1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, :, :, 0] = cv2.LUT(np.array(imgs[i, :, :, 0], dtype = np.uint8), table)
    return new_imgs


#===== 图像增强预处理过程 =====#
# (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape) == 4)  #4D arrays
    assert (data.shape[3] == 3)  # Use the original images
    # train_imgs = rgb2gray(data)
    train_imgs = extract_green_channel(data)  # the channel is 1
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.  # 将图片像素缩到0-1

    return train_imgs


# if __name__ == '__main__':
#     # b = np.array(Image.open("../../data/DRIVE/training/images/21_training.tif"))
#     # b = np.reshape(b, (1, b.shape[0], b.shape[1], 3))
#     # img_tmp = my_PreProc(b)
#     # img_tmp = np.reshape(img_tmp, (img_tmp.shape[1], img_tmp.shape[2]))
#     # img_tmp = Image.fromarray((img_tmp*255).astype(np.uint8))  # the image is between 0-1
#     # img_tmp.save("C:/Users/15016/Desktop/4.png")  # 保存合并后的图像
#     # print("图像预处理")