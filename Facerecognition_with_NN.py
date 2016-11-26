import time
import NeuralNetwork
import numpy as np
from lib import DisplayNetwork
from lib import Histogram
from lib import ImageFunctions
import matplotlib.pyplot as plt
from lib.TransferFunctions import sigmoid, linear
import commands
import os

__author__ = "Natasza Szczypien"

"""
    This code is loading face and nonface datas from the 'data' folder.
    The images are grey.
    The images are transformed to matrixes 19x19 pixel and then to a vector 1x361 [361 = 19x19]
    Next the faces and nonfaces are stacked in one list.
    This list is pushed to the backpropagation network.
    The calculation takes ~30 minutes.

"""

"""
This variables describes the folder with the Imagage data and the names of the images
"""
input_nodes = 361  # The images are transformed to matrixes 19x19 pixel and then to an array 1x361 [361 = 19x19]
hidden_nodes = 1600
output_nodes = 1  # true/falase or face/nonface

positives_path = 'data/LFaceData1600'
positive_name = 'face'
positives_amound = 1600
positives_test_path = 'data/LFaceData400'
positive_test_name = 'face'
positives_test_amound = 400

negatives_path = 'data/LNonfaceData1600'
negative_name = 'B'
negatives_amound = 10
negatives_test_path = 'data/LNonfaceData400'
negatives_test_amound = 400

file_name = '.jpg.jpg'

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)




def prepare_target_list(how_many, target_value):
    """
    Prepares the array with matrixes with the targets
    :param how_many: how many targets? = the output nodes
    :param target_value: what is the target value to learn
    :return: the target array: if target value is 1 and how_many is 4 => [[1],[1],[1],[1]]
    """
    target = []
    for one in range(how_many):
        target.append([target_value])
    target = np.array(target)
    return target


def check_output(output):
    """
    Checks if the output is a face or a nonface
    :param output:
    :return: a list of outputs
    """
    y = []
    for i in output:
        print "outuput", i
        if round(i) > 0:
            print "face"
        else:
            print "non-face"
        y.append(i)
    return y


def test_network(bpn, test_data):
    """
    Tests the network with test datas
    :return: the tested output
    """
    DisplayNetwork.display_green("[INFO] Started to test the network")
    output = bpn.Run(np.array(test_data))
    return output


def prepare_image_list(path, image_name, i_range):
    """
    This code is loading face and nonface datas from the 'data' folder.
    The images are grey.
    The images are transformed to matrices 19x19 pixel and then to a vector 1x361 [361 = 19x19]
    :param path:
    :type path:
    :param image_name:
    :return:
    """
    DisplayNetwork.display_green("[INFO] Loading the images to train the network")
    positives = []
    file_list = commands.getoutput('ls ./' + path + '/*.jpg | xargs -n 1 basename').split("\n")

    for i in i_range:
        image_name = path + '/' + file_list[i]
        DisplayNetwork.display_yellow("[INFO] Loading image" + image_name)

        image_matrix = ImageFunctions.turnImageToGray(image_name)  # Load image as gray
        reshaped = np.reshape(image_matrix, 361)  # makes 19x19 matrix to 1x361 vector
        positives.append(reshaped.tolist())

    return np.array(positives)


def getPreZero(i):
    """
    Because unfortunately the Images were saved as image_0000x.jpg.jpg
    :param i:
    :type i:
    :return:
    :rtype:
    """
    pre_zero = ''
    if bool(i % 10) != bool(i == 0):
        pre_zero = '0000'
    if round(((i % 100) / 10), 1):
        pre_zero = '000'
    if round(((i % 1000) / 100), 1):
        pre_zero = '00'
    if round(((i % 10000) / 1000), 1):
        pre_zero = '0'
    return pre_zero


def build_and_display_network():
    """
    Build the NN with NeuralNetwork.py
    Displays NN with DisplayNetwork.py
    :return: backpropagation network
    """
    bpn = NeuralNetwork.BackPropagationNetwork((input_nodes, hidden_nodes, output_nodes),[None, sigmoid, linear])
    DisplayNetwork.displayLayers(bpn.matrixDimension)

    return bpn

def start_face_recognition():
    start_time = time.time()
    print dir_path
    bpn = build_and_display_network()


    #-------------------------------------------------------------------------------
    """ Prepare list of images """
    faces = prepare_image_list(positives_path, positive_name, range(0, positives_amound))
    non_faces = prepare_image_list(negatives_path, negative_name, range(0, negatives_amound))

    """ Prepare the target """
    target_faces = prepare_target_list(len(faces), 1.0)
    target_non_faces = prepare_target_list(len(non_faces), -1.0)

    target = np.concatenate((target_faces, target_non_faces), axis=0)


    #-------------------------------------------------------------------------------
    """ Train the network """
    trainning_data = np.concatenate((faces,non_faces),axis=0)
    y = NeuralNetwork.trainNetwork(bpn, trainning_data, target)



    #-------------------------------------------------------------------------------
    """ Testing the network """

    """ TRAINING DATA """
    result_traning_faces = test_network(bpn, faces)
    Histogram.plot('Identification threshold', 'Output value for NN', 'Density', 'Face - training data', result_traning_faces)

    result_traning_nonfaces = test_network(bpn, non_faces)
    Histogram.plot('Identification threshold', 'Output value for NN', 'Density', 'Nonface - training data',
                   result_traning_nonfaces)

    """ TEST DATA"""

    #FACES
    test_faces = prepare_image_list(positives_test_path, positive_test_name,
                                    range(0, positives_test_amound))
    result_faces = test_network(bpn, test_faces)

    Histogram.plot('Identification threshold', 'Output value for NN','Density','Face - test data',  result_faces)

    # NONFACES
    test_non_faces = prepare_image_list(negatives_test_path, negative_name,
                                        range(0, negatives_test_amound))
    result_nonfaces = test_network(bpn, test_non_faces)
    Histogram.plot('Identification threshold', 'Output value for NN', 'Density', 'Nonface - test data', result_nonfaces)

    print "Execution time in seconds", time.time() - start_time
