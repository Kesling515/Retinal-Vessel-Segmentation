#===========================================================#
#                                                           #
#        将 DRIVE 数 据 集 转 为 hdf5 文 件                  #
#        读 取 hdf5 文 件                                   #
#                                                           #
#===========================================================#

import h5py
import numpy as np
import os.path
from PIL import Image
from glob import glob
from skimage.transform import resize


raw_training_x_path_DRIVE = './data/DRIVE/training/images/*.tif'
raw_training_y_path_DRIVE = './data/DRIVE/training/1st_manual/*.gif'
raw_training_mask_path_DRIVE = './data/DRIVE/training/mask/*.gif'
raw_test_x_path_DRIVE = './data/DRIVE/test/images/*.tif'
raw_test_y_path_DRIVE = './data/DRIVE/test/1st_manual/*.gif'
raw_test_mask_path_DRIVE = './data/DRIVE/test/mask/*.gif'

raw_data_path = None
raw_data_path_DRIVE = [ raw_training_x_path_DRIVE,
                        raw_training_y_path_DRIVE,
                        raw_training_mask_path_DRIVE,
                        raw_test_x_path_DRIVE,
                        raw_test_y_path_DRIVE,
                        raw_test_mask_path_DRIVE ]

HDF5_data_path = './data/HDF5/'

DESIRED_DATA_SHAPE_DRIVE = (576, 576)  #(584, 565) ==> (576, 576)
DESIRED_DATA_SHAPE = None


def isHDF5exists(raw_data_path, HDF5_data_path):
    for raw in raw_data_path:
        if not raw:
            continue
        raw_splited = raw.split('/')
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/*.hdf5'])
        # glob函数返回HDF5下所有hdf5文件
        if len(glob(HDF5)) == 0:
            return False
    return True


def read_input(path):
    if path.find('manual') > 0 or path.find('mask') > 0:
        x = np.array(Image.open(path)) / 255.
    else:
        x = np.array(Image.open(path)) / 255.
    if x.shape[-1] == 3:
        return x
    else:
        return x[..., np.newaxis]


def preprocessData(data_path, dataset):
    global DESIRED_DATA_SHAPE

    data_path = list(sorted(glob(data_path)))  #获取data_path下所有的文件名
    if data_path[0].find('mask') > 0:
        return np.array([read_input(image_path) for image_path in data_path])
    else:
        return np.array([resize(read_input(image_path), DESIRED_DATA_SHAPE) for image_path in data_path])


def createHDF5(data, HDF5_data_path):
    try:
        os.makedirs(HDF5_data_path, exist_ok=True)
    except:
        pass
    print("saving datasets:", HDF5_data_path ,"data.hdf5")
    with h5py.File(HDF5_data_path + 'data.hdf5', "w") as f:
        f.create_dataset("data", data = data, dtype = data.dtype)
    return


# 将DRIVE数据集训练数据和测试数据保存成hdf5文件
def prepareDataset(dataset):
    global raw_data_path, HDF5_data_path, raw_data_path_DRIVE
    global DESIRED_DATA_SHAPE

    DESIRED_DATA_SHAPE = DESIRED_DATA_SHAPE_DRIVE
    raw_data_path = raw_data_path_DRIVE

    print("*"*5, "saving train and test datasets", "*"*5)

    if isHDF5exists(raw_data_path, HDF5_data_path):
        print("HDF5 files already exist")
        return

    for raw in raw_data_path:
        if not raw:
            continue
        raw_splited = raw.split('/')
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/'])
        preprocessed = preprocessData(raw, dataset)
        createHDF5(preprocessed, HDF5)


# 获取训练数据集
def getTrainingData(XorY, dataset):
    global HDF5_data_path, raw_data_path_DRIVE

    if dataset == 'DRIVE':
        raw_training_x_path, raw_training_y_path, raw_training_mask_path = raw_data_path_DRIVE[:3]

    if XorY == 0:
        raw_splited = raw_training_x_path.split('/') # original images
    elif XorY == 1:
        raw_splited = raw_training_y_path.split('/') # groundtruth images
    else:
        if not raw_training_mask_path:
            return None
        raw_splited = raw_training_mask_path.split('/') # border masks

    data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    # print(data_path)
    with h5py.File(data_path, "r") as f:
        data = f["data"][()]

    return data


# 获取预测数据集
def getTestData(XorYorMask, dataset):
    global HDF5_data_path, raw_data_path_DRIVE

    if dataset == 'DRIVE':
        raw_test_x_path, raw_test_y_path, raw_test_mask_path = raw_data_path_DRIVE[3:]

    if XorYorMask == 0:
        raw_splited = raw_test_x_path.split('/') # original images
    elif XorYorMask == 1:
        raw_splited = raw_test_y_path.split('/') # groundtruth images
    else:
        if not raw_test_mask_path:
            return None
        raw_splited = raw_test_mask_path.split('/') # border masks

    data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    with h5py.File(data_path, "r") as f:
        data = f["data"][()]

    return data