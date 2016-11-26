import Facerecognition_with_NN
from NeuralNetwork import *
from lib import ORNetwork
from lib import XORNetwork
from lib.Enum import *
from lib.LogisticMap import *
from lib.Plot import XOR_OR_LogisticMap

__author__ = "Natasza Szczypien"

def logicNetwork(decision):
    """
    Boolean function that checks the decision of the user
    :param decision: users decision
    :return: true or false
    """
    return xor(decision == 2, decision == 3)

if __name__ == '__main__':
    """
    The main method to start the networks. You have this possibilities.

    First step: Define the network. Type a number:
    [0] Build your own network
    [1] Logistic map function
    [2] OR
    [3] XOR
    [4] Face recognition
    """

    print "\nFirst step: Define the network. Type a number: "
    decision = input(" [0] Build your own network\n [1] Logistic map function\n [2] OR\n [3] XOR\n [4] Face recognition")

    if (decision == FACE_RECOGNITION):
        Facerecognition_with_NN.start_face_recognition()

    else:

        if decision == LOGISTIC_MAP:

            inputNodes = 1
            hiddenNodes = 300
            outputNodes = 1

            initial_value = 0.41
            n = 100
            logisticMap_input = getValues(initial_value, n)
            logisticMap_target = getValues(initial_value, n)

            lastElement = len(logisticMap_input) - 1
            del logisticMap_input[lastElement]
            _lvInput = []

            for i in logisticMap_input:
                _lvInput.append([i])

            _lvInput = np.array(_lvInput)

            """ Delete first element """
            logisticMap_target.pop(0)
            target = logisticMap_target
            _target = []

            for i in target:
                _target.append([i])

            _target = np.array(_target)
            x = range(1, n, 1)



        else:
            _sigmo = True
            if logicNetwork(decision):

                if (decision == XOR):
                    _network = XORNetwork.XORNetwork()

                elif (decision == OR):
                    _network = ORNetwork.ORNetwork()

                _target = _network.target
                _lvInput = _network.lvInput
                inputNodes = _network.inputNodes
                hiddenNodes = _network.hiddenNodes
                outputNodes = _network.outputNodes
                x = _network.x

            elif decision == OWN_NETWORK:
                inputNodes = input("How many input nodes?")
                hiddenNodes = input("How many hidden nodes?")
                outputNodes = input("How many output nodes?")
                _lvInput = feeding(inputNodes)
                _target = np.array(input("Define the target as [[number_1], ..., [number_n]]"))
                x = _lvInput


        lvOutput = start_NN(inputNodes, hiddenNodes, outputNodes, _lvInput, _target)

        print "\nInput"
        displayVector(_lvInput)
        print "\nOutput"
        displayVector(lvOutput)
        XOR_OR_LogisticMap(x, lvOutput, decision)



