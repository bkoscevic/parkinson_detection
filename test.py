import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path
import pandas as pd

from utils import cf_matrix


def test():
    img_width = 0
    img_height = 0
    report_path = ''
    df_test_images = pd.DataFrame()
    data_dir = Path(r'dataset')

    best_model = r'spiral/best_model_spiral.hdf5'
    #best_model = r'wave/best_model_wave.hdf5'
    model = load_model(best_model)

    if 'spiral' in best_model:
        df_test_images = pd.DataFrame({'path': list(data_dir.glob(r'spiral/testing/*/*.png'))})
        report_path = 'spiral/spiral_report.txt'
        img_height = 128
        img_width = 128
    elif 'wave' in best_model:
        df_test_images = pd.DataFrame({'path': list(data_dir.glob(r'wave/testing/*/*.png'))})
        report_path = 'wave/wave_report.txt'
        img_height = 128
        img_width = 256

    df_test_images['label'] = df_test_images['path'].map(lambda x: x.parent.stem)
    df_test_images['path'] = df_test_images['path'].astype(str)
    df_test_images = df_test_images.sample(frac=1)

    batch_size = len(df_test_images['label'])

    data_generator = ImageDataGenerator()
    test_generator = data_generator.flow_from_dataframe(df_test_images,
                                                        x_col='path',
                                                        y_col='label',
                                                        color_mode="grayscale",
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        shuffle=False)

    test_generator.reset()
    probs = model.predict(test_generator, verbose=1)
    y_pred = np.where(probs > 0.5, 1, 0)

    y_true = df_test_images['label'].to_numpy().ravel()
    y_true = LabelBinarizer().fit_transform(y_true).ravel()

    class_names = [key for key in test_generator.class_indices.keys()]

    y_pred = y_pred.argmax(axis=1)
    print(y_pred)
    print(y_true)
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(report_path, 'w') as f:
        f.write(report)
    print(classification_report(y_true, y_pred, target_names=class_names))

    matrix = cf_matrix(y_true, y_pred)
    print(matrix)

test()
