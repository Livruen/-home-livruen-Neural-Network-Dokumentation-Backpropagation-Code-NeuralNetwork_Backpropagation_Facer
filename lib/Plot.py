import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.Enum import LOGISTIC_MAP
from lib.LogisticMap import *

#
# Plot helper
#

__author__ = "Natasza Szczypien"

def XOR_OR_LogisticMap(x, y, decision):
    """
    Plots the result of XOR OR and Logistic Map
    :param x:
    :param y:
    :param decision: decides how to plot
    """
    if (decision == LOGISTIC_MAP):
        plotLogisticMap(x, y)
        showPlot()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, y, c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

def error(y, text1, text2):
    """
    Plots the Error
    """
    plt.xlabel(text1)
    plt.ylabel(text2)
    plt.plot(y)
    plt.show()
