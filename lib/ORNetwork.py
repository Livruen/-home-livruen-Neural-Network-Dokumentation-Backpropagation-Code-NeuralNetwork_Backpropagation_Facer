import numpy as np

__author__ = "Natasza Szczypien"


class ORNetwork():
    """
    Data to build an OR network
    """
    inputNodes = 2
    hiddenNodes = 10
    outputNodes = 1
    target = np.array([[0], [1], [1], [1]])
    lvInput = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])

    # for the plot
    x = [0, 1, 0, 1]
    y = [0, 1, 1, 0]

    def __init__(self):
        pass
