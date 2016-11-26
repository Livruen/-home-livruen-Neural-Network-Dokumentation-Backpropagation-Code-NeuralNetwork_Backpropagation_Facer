#
# Imports
#
import matplotlib.pyplot as plt

"""
    calculates the logistic map function and plots it.
"""
__author__ = "Natasza Szczypien"

def logisticMap(x, a, n, listX):
    if n == 0:
        return x
    elif ((a <= 4) | (a >= 0)) & ((x <= 1.0) | (x >= 0.0)):
        listX.append(x % 100)
        x = a * x * (1 - x)

        return logisticMap(x, a, n - 1, listX)

    #
    """
    Returns: the list with the calculated values of logistic map
    """


#
def getValues(initial_value, n):
    listX = []
    logisticMap(initial_value, 4, n, listX)
    return listX

    #
    """
    Adds a function to the plot
    """


#
def plotLogisticMap(x, y):
    plt.plot(x, y)

    #
    """
    Shows every function that was added
    """


#
def showPlot():
    plt.show()

    #
    """
    if run as a script, create a test object

    """


#
if __name__ == '__main__':
    n = 100
    initial_value = 0.41
    values0 = getValues(initial_value, n)
    initial_value = 0.4
    values1 = getValues(initial_value, n)

    x = range(n)
    plotLogisticMap(x, values0)
    plotLogisticMap(x, values1)
    showPlot()


