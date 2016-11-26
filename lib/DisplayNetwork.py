#
# Imports
#
import time
# For the matrixes
import numpy as np
from lib.termcolor.termcolor import colored, cprint

"""
    Helper for displaying the NN elements
"""

__author__ = "Natasza Szczypien"



def displayVector(vector):
    """
    Displays a vector
    """
    index = 1
    for sample in np.nditer(vector):
        print  index, sample
        index += 1



def displayNN(matrix):
    """
    Display method for the neural network or some parts of it

    Arguments:
              NN - matrix
    """	    
    display_green(("[INFO] Building the network. Please wait ..."))
    print

    time.sleep(2)
    index = 1

    IH = matrix[0]
    HO = matrix[1]

    display_magenta(("{Input nodes} -> {Hidden nodes}"))
    print

    for i in IH:
        print 'Node ', index, i
        index += 1

    index = 1
    display_magenta(("{Output nodes} -> {Hidden nodes}"))
    print

    for j in HO:
        print 'Node ', index, j
        index += 1



def displayLayers(layers):
    """
    Display method for layers

    Arguments:
              layer vector
    """

    display_blue("\n_____________________ [ THE NEURAL NETWORK] _____________________\n")
    print
    display_blue(('INPUT NODES:'+ str(layers[0])+ ' HIDDEN NODES:'+str(layers[1])+ ' OUTPUT NODES:'+ str(layers[2])+ '\n'))


def display_green(text):
    """
    Displays a text in color green
    """
    print "\n\n"
    print colored(text, 'green', attrs=['reverse', 'blink'])

def display_blue(text):
    """
    Displays a text in color blue
    """
    print "\n\n"
    print colored(text, 'blue', attrs=['reverse', 'blink'])

def display_magenta(text):
    """
    Displays a text in color magenta
    """
    print "\n\n"
    print colored(text, 'magenta', attrs=['reverse', 'blink'])

def display_yellow(text):
    """
    Displays a text in color magenta
    """
    print colored(text, 'yellow', attrs=['reverse', 'blink'])

def display_cyan(text):
    """
    Displays a text in color magenta
    """
    print "\n\n"
    print colored(text, 'cyan', attrs=['reverse', 'blink'])
