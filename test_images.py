import json

import cv2
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

def plot_test_images(flag, parkinson, healthy):

    for_plot = []
    for elem in parkinson[:3]:
        for_plot.append(elem)
    for elem in healthy[:3]:
        for_plot.append(elem)

    plt.figure(figsize=(10, 10))
    if flag:
        plt.suptitle('Točno klasificirani primjeri iz skupa za testiranje')
        save_file = 'wave/wave_test_right.png'
    else:
        plt.suptitle('Pogrešno klasificirani primjeri iz skupa za testiranje')
        save_file = 'wave/wave_test_wrong.png'

    for i, element in enumerate(for_plot):
        label = element['label']
        prob = element['probability']
        img_path = element['picture']
        img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        print(label, prob, img_path)
        plt.subplot(2, 3, i+1)
        plt.title("{}\n probability: {:.2f}".format(label, float(prob)))
        plt.imshow(img)
        plt.savefig(save_file)
    plt.show()


if __name__ == '__main__':

    #file = 'spiral/spiral_results.json'
    file = 'wave/wave_results.json'

    file = open(file)
    data = json.load(file)

    healthy_right = []
    parkinson_right = []
    flag = True

    for elem in data:
        if elem['label'] == 'parkinson' and 'P' in elem['picture']:
            parkinson_right.append(elem)
        elif elem['label'] == 'healthy' and 'H' in elem['picture']:
            healthy_right.append(elem)

    plot_test_images(flag, parkinson_right, healthy_right)

    healthy_wrong = []
    parkinson_wrong = []
    flag = False

    for elem in data:
        if elem['label'] == 'parkinson' and 'P' not in elem['picture']:
            parkinson_wrong.append(elem)
        elif elem['label'] == 'healthy' and 'H' not in elem['picture']:
            healthy_wrong.append(elem)

    plot_test_images(flag, parkinson_wrong, healthy_wrong)

