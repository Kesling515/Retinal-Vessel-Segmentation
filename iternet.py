#===========================================================#
#                                                           #
#     - Load the images and extract the patches             #
#     - Define the neural network model                     #
#     - Define the training                                 #
#     - Define the prediction                               #
#     - Model name: IterNet                                 #
#                                                           #
#===========================================================#


import numpy as np
import os
import cv2
import os.path
import pickle
from datetime import datetime
import threading
from keras import losses
from tqdm import tqdm
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from random import randint
from skimage.transform import resize
from keras.layers import Input, MaxPooling2D, concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU
from keras.models import Model
from keras.optimizers import Adam
from utils.evaluate import evaluate
from utils import prepare_dataset, data_augmentation, pre_processing, crop_prediction


# 切片器：提取宽高为 crop_size*crop_size 的patch
def random_crop(img, mask, crop_size):
    imgheight = img.shape[0]
    imgwidth = img.shape[1]

    i = randint(0, imgheight - crop_size)
    j = randint(0, imgwidth - crop_size)

    return img[i:(i + crop_size), j:(j + crop_size), :], mask[i:(i + crop_size), j:(j + crop_size)]


# 数据生成器：每次生成一个batch数据
class Generator():
    def __init__(self, batch_size, repeat, dataset):
        self.lock = threading.Lock()
        self.dataset = dataset
        with self.lock:
            self.list_images_all = prepare_dataset.getTrainingData(0, self.dataset)  #raw_training_x
            self.list_gt_all = prepare_dataset.getTrainingData(1, self.dataset)  #raw_training_y
        self.n = len(self.list_images_all)
        self.index = 0
        self.repeat = repeat
        self.batch_size = batch_size
        self.step = self.batch_size // self.repeat

        if self.repeat >= self.batch_size:
            self.repeat = self.batch_size
            self.step = 1

    def gen(self, au=True, crop_size=64, iteration=None):

        while True:
            data_yield = [self.index % self.n,
                          (self.index + self.step) % self.n if (self.index + self.step) < self.n else self.n]
            self.index = (self.index + self.step) % self.n

            list_images_base = self.list_images_all[data_yield[0]:data_yield[1]]
            list_gt_base = self.list_gt_all[data_yield[0]:data_yield[1]]

            list_images_base = pre_processing.my_PreProc(list_images_base)  # 图片增强预处理

            list_images_aug = []
            list_gt_aug = []
            for image, gt in zip(list_images_base, list_gt_base):
                if au:
                    if crop_size == prepare_dataset.DESIRED_DATA_SHAPE[0]:
                        for _ in range(self.repeat):
                            image, gt = data_augmentation.random_augmentation(image, gt)
                            list_images_aug.append(image)
                            list_gt_aug.append(gt)
                    else:
                        image, gt = data_augmentation.random_augmentation(image, gt)
                        list_images_aug.append(image)
                        list_gt_aug.append(gt)
                else:
                    list_images_aug.append(image)
                    list_gt_aug.append(gt)

            list_images = []
            list_gt = []

            if crop_size == prepare_dataset.DESIRED_DATA_SHAPE[0]:
                list_images = list_images_aug
                list_gt = list_gt_aug
            else:
                for image, gt in zip(list_images_aug, list_gt_aug):
                    for _ in range(self.repeat):
                        image_, gt_ = random_crop(image, gt, crop_size)

                        list_images.append(image_)
                        list_gt.append(gt_)
            outs = {}
            for iteration_id in range(iteration):
                outs.update({f'out1{iteration_id + 1}': np.array(list_gt)})
            outs.update({'final_out': np.array(list_gt)})

            yield np.array(list_images), outs


# 定义网络模型：IterNet
def get_iternet(minimum_kernel=32, do=0, crop_size=64, activation=ReLU, iteration=3):
    inputs = Input((crop_size, crop_size, 1))
    # encoding path
    conv1 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(inputs)))
    conv1 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(conv1)))
    a = conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(pool1)))
    conv2 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv2)))
    b = conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(pool2)))
    conv3 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(pool3)))
    conv4 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Dropout(do)(activation()(Conv2D(minimum_kernel * 16, (3, 3), padding='same')(pool4)))
    conv5 = Dropout(do)(activation()(Conv2D(minimum_kernel * 16, (3, 3), padding='same')(conv5)))

    # decoding + concat path
    up6 = concatenate([Conv2DTranspose(minimum_kernel * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4],
                      axis=3)
    conv6 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(up6)))
    conv6 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv6)))

    up7 = concatenate([Conv2DTranspose(minimum_kernel * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3],
                      axis=3)
    conv7 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(up7)))
    conv7 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv7)))

    up8 = concatenate([Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2],
                      axis=3)
    conv8 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(up8)))
    conv8 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv8)))

    up9 = concatenate([Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')(conv8), conv1],
                      axis=3)
    conv9 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(up9)))
    conv9 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(conv9)))

    pt_conv1a = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation1a = activation()
    pt_dropout1a = Dropout(do)
    pt_conv1b = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation1b = activation()
    pt_dropout1b = Dropout(do)
    pt_pooling1 = MaxPooling2D(pool_size=(2, 2))

    pt_conv2a = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation2a = activation()
    pt_dropout2a = Dropout(do)
    pt_conv2b = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation2b = activation()
    pt_dropout2b = Dropout(do)
    pt_pooling2 = MaxPooling2D(pool_size=(2, 2))

    pt_conv3a = Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_activation3a = activation()
    pt_dropout3a = Dropout(do)
    pt_conv3b = Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_activation3b = activation()
    pt_dropout3b = Dropout(do)

    pt_tranconv8 = Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')
    pt_conv8a = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation8a = activation()
    pt_dropout8a = Dropout(do)
    pt_conv8b = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation8b = activation()
    pt_dropout8b = Dropout(do)

    pt_tranconv9 = Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')
    pt_conv9a = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation9a = activation()
    pt_dropout9a = Dropout(do)
    pt_conv9b = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation9b = activation()
    pt_dropout9b = Dropout(do)

    conv9s = [conv9]
    outs = []
    a_layers = [a]
    for iteration_id in range(iteration):
        out = Conv2D(1, (1, 1), activation='sigmoid', name=f'out1{iteration_id + 1}')(conv9s[-1])
        outs.append(out)

        model = Model(inputs=[inputs], outputs=[outs[-1]])
        count = 0
        for i, layer in enumerate(model.layers):
            if not layer.name.startswith('out'):
                count += 1

        conv1 = pt_dropout1a(pt_activation1a(pt_conv1a(conv9s[-1])))
        conv1 = pt_dropout1b(pt_activation1b(pt_conv1b(conv1)))
        a_layers.append(conv1)
        conv1 = concatenate(a_layers, axis=3)
        conv1 = Conv2D(minimum_kernel, (1, 1), padding='same')(conv1)
        pool1 = pt_pooling1(conv1)

        conv2 = pt_dropout2a(pt_activation2a(pt_conv2a(pool1)))
        conv2 = pt_dropout2b(pt_activation2b(pt_conv2b(conv2)))
        pool2 = pt_pooling2(conv2)

        conv3 = pt_dropout3a(pt_activation3a(pt_conv3a(pool2)))
        conv3 = pt_dropout3b(pt_activation3b(pt_conv3b(conv3)))

        up8 = concatenate([pt_tranconv8(conv3), conv2], axis=3)
        conv8 = pt_dropout8a(pt_activation8a(pt_conv8a(up8)))
        conv8 = pt_dropout8b(pt_activation8b(pt_conv8b(conv8)))

        up9 = concatenate([pt_tranconv9(conv8), conv1], axis=3)
        conv9 = pt_dropout9a(pt_activation9a(pt_conv9a(up9)))
        conv9 = pt_dropout9b(pt_activation9b(pt_conv9b(conv9)))

        conv9s.append(conv9)

    out2 = Conv2D(1, (1, 1), activation='sigmoid', name='final_out')(conv9)
    outs.append(out2)

    model = Model(inputs=[inputs], outputs=outs)
    model.summary()

    count = 0
    for i, layer in enumerate(model.layers):
        if not layer.name.startswith('model1_') and not layer.name.startswith('out'):
            layer.name = layer.name[:layer.name.rfind('_')]
            layer.name = f'model2_{layer.name}_{count}'
            count += 1

    loss_funcs = {}
    for iteration_id in range(iteration):
        loss_funcs.update({f'out1{iteration_id + 1}': losses.binary_crossentropy})
    loss_funcs.update({'final_out': losses.binary_crossentropy})

    metrics = {
        "final_out": ['accuracy']
    }

    model.compile(optimizer=Adam(lr=1e-3), loss=loss_funcs, metrics=metrics)

    return model


# 训练函数：进行预处理数据集、构造编译模型、训练模型、保存模型
def train(DATASET="DRIVE", crop_size=64, need_au=True, ACTIVATION='ReLU', dropout=0.2, batch_size=20,
          repeat=5, minimum_kernel=32, epochs=50, iteration=3):

    print('-'*40)
    print('Loading and preprocessing train data...')
    print('-'*40)

    network_name = "IterNet"
    model_name = f"{network_name}_cropsize_{crop_size}_epochs_{epochs}"
    print("Model : %s" % model_name)
    prepare_dataset.prepareDataset(DATASET)
    activation = globals()[ACTIVATION]

    print('-' * 35)
    print('Creating and compiling model...')
    print('-' * 35)
    model = get_iternet(minimum_kernel=minimum_kernel, do=dropout, crop_size=crop_size, activation=activation,
                        iteration=iteration)

    try:
        os.makedirs(f"./trained_model/{model_name}/", exist_ok=True)
        os.makedirs(f"./logs/{model_name}/", exist_ok=True)
    except:
        pass

    plot_model(model, to_file = './trained_model/'+ model_name + '/'+ model_name + '_model.png')   #check how the model looks like
    json_string = model.to_json()
    with open('./trained_model/'+ model_name + '/'+ model_name + '_architecture.json', 'w') as jsonfile:
        jsonfile.write(json_string)

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d---%H-%M-%S")

    tensorboard = TensorBoard(
        log_dir=f"./logs/{model_name}/{model_name}---{date_time}",
        histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
        write_images=True, embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    save_path = f"./trained_model/{model_name}/{model_name}.hdf5"
    checkpoint = ModelCheckpoint(save_path, monitor='final_out_loss', verbose=1, save_best_only=True, mode='min')

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    data_generator = Generator(batch_size, repeat, DATASET)
    history = model.fit_generator(data_generator.gen(au=need_au, crop_size=crop_size, iteration=iteration),
                                  epochs=epochs, verbose=1,
                                  steps_per_epoch=1000 * data_generator.n // batch_size,
                                  use_multiprocessing=False, workers=0,
                                  callbacks=[tensorboard, checkpoint])

    print('-'*30)
    print('Training finished')
    print('-'*30)


# 预测函数：加载测试数据集、加载权重、预测数据、保存测试结果
def predict(ACTIVATION='ReLU', dropout=0.2, minimum_kernel=32,
            epochs=50, crop_size=64, stride_size=3, iteration=3, DATASET='DRIVE'):

    print('-'*40)
    print('Loading and preprocessing test data...')
    print('-'*40)

    network_name = "IterNet"
    model_name = f"{network_name}_cropsize_{crop_size}_epochs_{epochs}"

    prepare_dataset.prepareDataset(DATASET)
    test_data = [prepare_dataset.getTestData(0, DATASET),
                 prepare_dataset.getTestData(1, DATASET),
                 prepare_dataset.getTestData(2, DATASET)]
    IMAGE_SIZE = None
    if DATASET == 'DRIVE':
        IMAGE_SIZE = (565, 584)

    gt_list_out = {}
    pred_list_out = {}
    for out_id in range(iteration + 1):
        try:
            os.makedirs(f"./output/{model_name}/crop_size_{crop_size}/out{out_id + 1}/", exist_ok=True)
            gt_list_out.update({f"out{out_id + 1}": []})
            pred_list_out.update({f"out{out_id + 1}": []})
        except:
            pass

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    activation = globals()[ACTIVATION]
    model = get_iternet(minimum_kernel=minimum_kernel, do=dropout, crop_size=crop_size, activation=activation,
                        iteration=iteration)
    print("Model : %s" % model_name)
    load_path = f"./trained_model/{model_name}/{model_name}.hdf5"
    model.load_weights(load_path, by_name=False)

    imgs = test_data[0]
    segs = test_data[1]
    masks = test_data[2]

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    print('\n')

    for i in tqdm(range(len(imgs))):

        img = imgs[i]  # (576,576,3)
        seg = segs[i]  # (576,576,1)
        mask = masks[i]  # (584,565,1)

        patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(img, crop_size, stride_size)
        preds = model.predict(patches_pred)  # 预测数据

        out_id = 0
        for pred in preds:
            pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
            pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
            pred_imgs = pred_imgs[:, 0:prepare_dataset.DESIRED_DATA_SHAPE[0], 0:prepare_dataset.DESIRED_DATA_SHAPE[0], :]
            probResult = pred_imgs[0, :, :, 0]  # (576,576)
            pred_ = probResult
            with open(f"./output/{model_name}/crop_size_{crop_size}/out{out_id + 1}/{i + 1:02}.pickle", 'wb') as handle:
                pickle.dump(pred_, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pred_ = resize(pred_, IMAGE_SIZE[::-1])  # (584,565)
            mask_ = mask
            mask_ = resize(mask_, IMAGE_SIZE[::-1])  # (584,565)
            seg_ = seg
            seg_ = resize(seg_, IMAGE_SIZE[::-1])  # (584,565)
            gt_ = (seg_ > 0.5).astype(int)
            gt_flat = []
            pred_flat = []
            for p in range(pred_.shape[0]):
                for q in range(pred_.shape[1]):
                    if mask_[p, q] > 0.5:  # Inside the mask pixels only
                        gt_flat.append(gt_[p, q])
                        pred_flat.append(pred_[p, q])

            gt_list_out[f"out{out_id + 1}"] += gt_flat
            pred_list_out[f"out{out_id + 1}"] += pred_flat

            pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
            cv2.imwrite(f"./output/{model_name}/crop_size_{crop_size}/out{out_id + 1}/{i + 1:02}.png", pred_)
            out_id += 1

    print('-'*30)
    print('Prediction finished')
    print('-'*30)
    print('\n')

    print('-'*30)
    print('Evaluate the results')
    print('-'*30)

    for out_id in range(iteration + 1)[-1:]:
        print('\n\n', f"out{out_id + 1}")
        evaluate(gt_list_out[f"out{out_id + 1}"], pred_list_out[f"out{out_id + 1}"], DATASET, epochs,
                 crop_size, network_name)

    print('-'*30)
    print('Evaluate finished')
    print('-'*30)


if __name__ == "__main__":
    train(DATASET="DRIVE", iteration=3, crop_size=64, batch_size=20, epochs=5)
    predict(DATASET='DRIVE', iteration=3, crop_size=64, stride_size=3, epochs=5)