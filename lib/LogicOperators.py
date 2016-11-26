__author__ = "Natasza Szczypien"

class LogicOperators(object):
    inputNodes = 2
    hiddenNodes = 10
    outputNodes = 1
    target = 0

    def __init__(self, target):
        self.target = target

    def target(self):
        return self.target

    def inputNodes(self):
        return self.inputNodes

    def hiddenNodes(self):
        return self.hiddenNodes

    def outputNodes(self):
        return self.outputNodes
