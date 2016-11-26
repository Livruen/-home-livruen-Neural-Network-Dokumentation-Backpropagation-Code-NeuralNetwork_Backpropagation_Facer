import matplotlib.pyplot as plt
from numpy.random import normal, uniform, np

__author__ = "Natasza Szczypien"


def plot(title, xlabel, ylabel, name, image):
    """
    Plots a histogram for Facerecognition_with_NN.py
    """
    print image
    plt.hist(image, bins=20, histtype='stepfilled', facecolor='green', label=name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([-1, 2, 0, 10])
    plt.legend()
    plt.show()
