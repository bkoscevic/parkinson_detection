from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
from PIL import Image
from utils import copy_files

if not os.path.exists('dataset/wave/training/augm_healthy/'):
    os.makedirs('dataset/wave/training/augm_healthy/')

if not os.path.exists('dataset/wave/training/augm_parkinson/'):
    os.mkdir('dataset/wave/training/augm_parkinson/')

healthy_directory = 'dataset/wave/training/healthy/'
parkinson_directory = 'dataset/wave/training/parkinson/'

healthy_save = 'dataset/wave/training/augm_healthy/'
parkinson_save = 'dataset/wave/training/augm_parkinson/'



def augmentation(dir_original, dir_save):
    data_generator = ImageDataGenerator(rotation_range=90,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        brightness_range=(0.5, 1.5),
                                        horizontal_flip=True,
                                        vertical_flip=True, )

    img_height = 256
    img_width = 256
    dataset = []
    my_images = os.listdir(dir_original)
    for i, image_name in enumerate(my_images):
        if (image_name.split('.')[1] == 'png'):
            image = io.imread(dir_original + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((img_width, img_height))
            dataset.append(np.array(image))
    x = np.array(dataset)
    i = 0
    for batch in data_generator.flow(x, batch_size=64,
                                     save_to_dir=dir_save,
                                     save_prefix='AUGM',
                                     save_format='png'):
        i += 1
        if i > 100:
            break

    print('Finished!')


augmentation(healthy_directory, healthy_save)
augmentation(parkinson_directory, parkinson_save)


copy_files(healthy_directory, healthy_save)
copy_files(parkinson_directory, parkinson_save)
