#
#  Imports
#
from operator import xor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib import DisplayNetwork
from lib import Plot
from lib.DisplayNetwork import *
from lib.Enum import LOGISTIC_MAP
from lib.LogisticMap import *
from lib.TransferFunctions import linear, sigmoid

"""
   The vast majority of the code comes from a youtube video: https://www.youtube.com/watch?v=XqRUHEeiyCs
   The Coder name is Ryan Harris.
   I made some more comments and addet methods to make the code cleaner for me.
   """
__author__ = "Natasza Szczypien"



class BackPropagationNetwork:

    # Class members
    layerCount = 0  # All layers without hidden layer
    matrixDimension = None  # Like (1, 3)
    weights = []  # Weights in Adjacency matrix shape
    tFuncs = []  # Transfer function list

    # Class methods
    def __init__(self, network, layerFunction=None):
        """ Initialize the network """

        """ Layer info """
        self.layerCount = len(network) - 1
        self.matrixDimension = network

        if layerFunction is None:
            lFuncs = []
            for i in range(self.layerCount):
                if i == self.layerCount - 1:
                    lFuncs.append(linear)
                else:
                    lFuncs.append(sigmoid)
        else:
            if len(network) != len(layerFunction):
                raise ValueError("Incopatible list of transfer functions")
            elif (layerFunction[0] is not None):
                raise ValueError("Input layer cannot have a transfer fuction")
            else:
                lFuncs = layerFunction[1:]

        self.tFuncs = lFuncs

        """ Data from last Run """
        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []

        """ Create the weight arrays
        #
        # Example: layerSize = (1, 2, 1) <=>
        #
        #  IL  HL  OL
        #  O - O - O
        #    \   /
        #      O
        """

        """ Building two Adjacency matrixes separeted in the Hidden layer """
        for (layer1, layer2) in zip(network[:-1], network[1:]):
            randomWeight = np.random.normal(scale=0.01, size=(layer2, layer1 + 1))  # """ +1 for BIOS node """
            self.weights.append(randomWeight)
            self._previousWeightDelta.append(np.zeros((layer2, layer1 + 1)))

    def last(self):
        """
        Last element from List
        :return: last element
        """
        return -1

    def first(self):
        """
        First element from List (to make the code cleaner)
        :return:First element
        """
        return 0


        return food

    def Run(self, input):
        """
         Run the network based on the network data
        :param input: The input for the network
        :return: The network output
        """


        """ Number of input cases (Input Layer and Hidden Layers) """
        lnCases = input.shape[0]  # n from dimension of input vector (n x m)

        """ Clear the previous intermediate value lists """
        self._layerInput = []
        self._layerOutput = []

        for index in range(self.layerCount):
            # Determine layer input
            if index == 0:

                """ Makes fake inputs for the BIOS Nodes """
                matrixOfOnes = np.ones([1, lnCases])

                """ Makes the input as a column """
                transposeInput = input.T

                """ Take a sequence of arrays and stack them vertically to make a single array """
                stackInput = np.vstack([transposeInput, matrixOfOnes])

                """ dot = multiplikation function """
                """ Multiplies the weights with the input """
                layerInput = self.weights[self.first()].dot(stackInput)

            else:

                """ Makes fake inputs for the BIOS Nodes """
                matrixOfOnes = np.ones([1, lnCases])

                """ Array[-1] means take the last element from list  """
                outputFromPreviousLayer = self._layerOutput[self.last()]
                stackInput = np.vstack([outputFromPreviousLayer, matrixOfOnes])
                layerInput = self.weights[index].dot(stackInput)

            """ Save to a global variable """
            self._layerInput.append(layerInput)

            """ Send every output through the activation function """
            self._layerOutput.append(self.tFuncs[index](layerInput))
            output = self._layerOutput[self.last()].T

        return output

    def TrainEpoch(self, lvInput, target, trainingRate=0.000001, momentum=0.2):
        """
         The real training method for NN for one Epoch,
         We changed the trainingRate and momentum
        :param lvInput:
        :param target:
        :param trainingRate:
        :param momentum:
        :return: error value

        """

        delta = []

        """ Number of input cases (Input Layer and Hidden Layers) """
        lnCases = lvInput.shape[0]  # n from dimension of input vector (n x m)

        # First run the Network
        self.Run(lvInput)

        # Calculate our deltas
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:  # Output layer

                # Compare to the target values
                output_delta = self._layerOutput[index] - target.T
                # a ** b = a^b
                error = np.sum(output_delta ** 2)
                # Calculating delta for each layer
                derivative = self.tFuncs[index](self._layerInput[index], True)
                delta.append(output_delta * derivative)
            else:
                # Compare to the following layer's delta
                delta_pullback = self.weights[index + 1].T.dot(delta[self.last()])
                derivative = self.tFuncs[index](self._layerInput[index], True)
                delta.append(delta_pullback[:self.last(), :] * derivative)

                # Compute weigh deltas
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index

            if index == 0:
                layerOutput = np.vstack([lvInput.T, np.ones([1, lnCases])])
            else:
                layerOutput = np.vstack(
                    [self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            curWeightDelta = np.sum(
                layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0), axis=0)

            weightDelta = trainingRate * curWeightDelta + momentum * self._previousWeightDelta[index]
            self.weights[index] -= weightDelta
            self._previousWeightDelta[index] = weightDelta

        return error



def trainNetwork(bpn, lvInput, target):
    """
    Trains the network and calculates the error
    :param bpn: Backpropagation class
    :param lvInput: the input for the network
    :param target: the target that the network should learn
    :return: the calculated output (should be like aproximetally to the target)
    """
    DisplayNetwork.display_green("Starting to train the network")

    global i, lvOutput
    lnMax = 10000
    lnErr = 1e-6
    y = []
    for i in range(lnMax + 1):

        err = bpn.TrainEpoch(lvInput, target)
        y.append(err)
        if (i % 2500 == 0):
            print("Iteration {0}\tError: {1:0.6f}".format(i, err))
        if err <= lnErr:
            print("Minimum error reached at iteration {0}".format(i))
            break
        else:
            print("Iteration {0}\tError: {1:0.6f}".format(i, err))
    lvOutput = bpn.Run(lvInput)
    Plot.error(y, 'Iteration', 'Error')

    return lvOutput


def feeding(inputNodes):
    """
    Generates a random input for the network
    :param inputNodes: number of input nodes
    :return: random input for the network
    """
    food = np.array(np.random.rand(1, inputNodes))
    print "\nFeeding input"
    displayVector(food)


def start_NN(inputNodes, hiddenNodes, outputNodes, lvInput, target):
    """
        Creates the NN with the Backpropagation class
        Displays the network
        Starts to train the network with the input and target
    :param inputNodes: number of input nodes
    :param hiddenNodes: number of hidden nodes
    :param outputNodes: number of output nodes
    :param lvInput: Input for the Network
    :param target: the target that the network should learn
    :return: lvOutput  (should be like aproximetally to the target)
    """
    bpn = BackPropagationNetwork((inputNodes, hiddenNodes, outputNodes))
    displayLayers(bpn.matrixDimension)
    displayNN(bpn.weights)
    lvOutput = trainNetwork(bpn, lvInput, target)
    return lvOutput

