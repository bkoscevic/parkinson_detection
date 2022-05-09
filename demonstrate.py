import json
import os
import random

import cv2
import numpy as np
import keras.backend
from keras.models import load_model


def demonstrate():
    result_path = ''
    test_path = r''
    img_width = 0
    img_height = 0

    #best_model = r'spiral/best_model_spiral.hdf5'
    best_model = r'wave/best_model_wave.hdf5'
    model = load_model(best_model)

    if 'spiral' in best_model:
        test_path = r'dataset/spiral/testing'
        result_path = 'spiral/spiral_results.json'
        img_height = 128
        img_width = 128
    elif 'wave' in best_model:
        test_path = r'dataset/wave/testing'
        result_path = 'wave/wave_results.json'
        img_height = 128
        img_width = 256

    class_names = ['healthy', 'parkinson']
    print(class_names)

    test_images = []

    for subdir in os.listdir(test_path):
        subdir_path = os.path.join(test_path, subdir)
        for file in os.listdir(os.path.join(subdir_path)):
            file_path = os.path.join(subdir_path, file)
            if os.path.isfile(file_path) and file.endswith('.png'):
                test_images.append(file_path)

    print(test_images)
    print(len(test_images))

    results = []

    for i, image_name in enumerate(test_images):
        filename = image_name

        img = cv2.imread(filename)
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_CUBIC)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_img = np.expand_dims(rgb_img, 0)
        prediction = model.predict(rgb_img)
        prob = np.max(prediction)
        class_id = np.argmax(prediction)
        print('Predict: {}; class: {}'.format(class_names[class_id], class_id))
        results.append({'label': class_names[class_id], 'probability': '{:.4}'.format(prob), 'picture': filename})
        cv2.imwrite('images/{}_out.png'.format(i), img)

    with open(result_path, 'w') as file:
        json.dump(results, file, indent=4)

    keras.backend.clear_session()

demonstrate()
