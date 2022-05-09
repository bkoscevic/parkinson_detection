import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def copy_files(src_dir, dest_dir):
    for file_name in os.listdir(src_dir):
        # construct full file path
        source = src_dir + file_name
        destination = dest_dir + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)

def cf_matrix(y_true, y_pred):

    cf_matrix = confusion_matrix(y_true, y_pred)

    ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix\n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    plt.savefig('cf_matrix.png')
    plt.show()

    return cf_matrix
