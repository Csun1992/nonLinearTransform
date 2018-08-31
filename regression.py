import numpy as np
import sys
from perceptron import * 

class Regression(Perceptron):
    def __init__(self, data, group, testData, testGroup, regParam):
        super(Regression, self).__init__(data, group)
        self.dataDim = np.size(data, 0)
        self.testSize = np.size(testData, 0)
        self.testData = np.concatenate((np.ones((self.testSize, 1)), testData), axis=1)
        self.testGroup = testGroup
        self.classifiedTest = []
        self.regParam = regParam

    def findLinCoeff(self):
        eye = np.diag(np.ones((self.dataDim, 1)))
        pseudoInv = np.linalg.inv(self.data.T.dot(self.data)+self.regParam*eye).dot(self.data.T)
        correctGroup = np.array(self.correctGroup).reshape(-1, 1)
        self.linCoeff = pseudoInv.dot(correctGroup)
        return self.linCoeff

    def classify(self):
        self.findLinCoeff()
        predictedVal = self.data.dot(self.linCoeff) 
        self.classifiedGroup = [sign(predictedVal[i]) for i in range(self.size)]
        return self.classifiedGroup

    def getMisclassifiedIndex(self):
        self.misclassifiedIndex = []
        for i in range(self.size):
           if(self.classifiedGroup[i] != self.correctGroup[i]):
              self.misclassifiedIndex.append(i) 
        return self.misclassifiedIndex

    def getMisclassified(self): 
        self.misclassified = [self.correctGroup[i] != self.classifiedGroup[i] for i in
        range(self.size)]
        return self.misclassified

    def getInSampleErr(self):
        return sum(self.misclassified)/float(self.size)

    def run(self):
        self.classify()
        self.getMisclassified()
        return self.getInSampleErr()
    
    def getLinCoeff(self):
        return self.linCoeff
   
    def classifyTest(self):
        for i in range(self.testSize):
            self.classifiedTest.append(sign(self.linCoeff.T.dot(self.testData[i, :])))
        return self.classifiedTest

    def getOutSampleErr(self):
        self.classifyTest()
        err = 0
        for (i, j) in zip(self.classifiedTest, self.testGroup):
            err += (i!=j)
        return float(err)/self.testSize
