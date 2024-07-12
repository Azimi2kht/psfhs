import glob
from itk import imread
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import os


folder_path = "/home/azimi2kht/projects/psfh/data/raw/PSFHS/"  # Add the path to your data directory


def load_data(img_height, img_width, images_to_be_loaded, dataset):
    IMAGES_PATH = folder_path + 'image_mha/'
    MASKS_PATH = folder_path + 'label_mha/'

    train_ids = []

    if dataset == 'psfhs':
        train_ids = glob.glob(IMAGES_PATH + "*.mha")

    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)

    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.uint8)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

    print('Resizing training images and masks: ' + str(images_to_be_loaded))
    for n, id_ in tqdm(enumerate(train_ids)):
        if n == images_to_be_loaded:
            break

        image_path = id_
        mask_path = image_path.replace("image", "label")

        image = np.asarray(imread(image_path)).transpose(1, 2, 0)
        mask = np.asarray(imread(mask_path))

        X_train[n] = image
       
        # mask_threshold = 127
        # mask[:, :] = np.where(mask_[:, :, 0] >= mask_threshold, 1, 0)

        Y_train[n] = mask

    Y_train = np.expand_dims(Y_train, axis=-1).squeeze()

    return X_train, Y_train
