import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace

plt.rcParams.update({'font.size': 20})

def plot_evaluation():

    filename = ''
    path = 'logs/spiral_training_best.log'
    #path = 'wave/wave_training_best.log'

    if 'spiral' in path:
        filename = 'spiral/spiral_'
    if 'wave' in path:
        filename = 'wave/wave_'

    file = open(path, 'r')
    data = list()
    for line in file.readlines():
        data.append(line.rstrip().split(','))
    data = np.asarray(data[1:])
    print(data)


    acc, loss, val_acc, val_loss = [], [], [], []
    log_elements = [acc, loss, val_acc, val_loss]
    i = 1
    for element in log_elements:
        for value in data[:, i]:
            element.append(float(value))
        i += 1

    print(log_elements)

    plt.figure(figsize=(10, 15))
    plt.title("Gubitak modela")
    plt.plot(log_elements[1], 'b', label='Gubitak treniranje')
    plt.plot(log_elements[3], 'r', label='Gubitak validacije')
    plt.legend()
    plt.xlabel('Epoha')
    plt.ylabel('Gubitak')
    plt.ylim([0,2])
    plt.savefig(filename + 'loss.png')
    plt.show()

    plt.figure(figsize=(10, 15))
    plt.title("Točnost modela")
    plt.plot(log_elements[0], 'b', label='Točnost treniranje')
    plt.plot(log_elements[2], 'r', label='Točnost validacije')
    plt.legend()
    plt.xlabel('Epoha')
    plt.ylabel('Točnost')
    plt.ylim([0.45, 1])
    plt.savefig(filename + 'acc.png')
    plt.show()

plot_evaluation()
