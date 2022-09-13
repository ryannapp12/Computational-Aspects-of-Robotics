import matplotlib.pyplot as plt
import math
import sys
from labelers.connected_component_labelers import *
import cv2

def main():
    imgs = []
    img = cv2.imread("problem1.png")
    imgs.append(img)
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        shape = img.shape
        print(shape)
        img.reshape((1, shape[0] * shape[1]))
        labeler = RecursiveConnectedComponentLabeler()
        img = np.reshape(img, (shape[0], shape[1]))
        print(img)
        labeled_img = labeler.label_components(img)

        fig = plt.figure()
        ax = fig.add_subplot(121)

        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title("Original Image")

        ax = fig.add_subplot(122)
        plt.imshow(labeled_img)
        plt.axis('off')
        plt.title("Labeled Image")

        for (j, i), label in np.ndenumerate(labeled_img):
            ax.text(i, j, int(label), ha='center', va='center')

        plt.show()
        del labeler


if __name__ == "__main__":
    main()
    exit()
