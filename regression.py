import numpy as np
import sys
from perceptron import * 

class Regression(Perceptron):
    def __init__(self, data, group):
        super(Regression, self).__init__(data, group)

    def findLinCoeff(self):
        pseudoInv = np.linalg.inv(self.data.T.dot(self.data)).dot(self.data.T)
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
    

if __name__ == "__main__":
    """
    experimentNum = 1000
    dataDim = 2
    lowerLim = -1
    upperLim = 1

    # experiment with sample size = 100
    sampleSize = 100
    inSampleErrs = []

    for i in range(experimentNum):
        data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
        x1 = data[0, :]
        x2 = data[1, :]
        slope, intercept = findLine(x1, x2)
        group = classify(data, slope, intercept)

        regression = Regression(data, group)
        inSampleErrs.append(regression.run())
    print "For sample size = 100, the average in sample error is:"
    print sum(inSampleErrs)/experimentNum

    # estimate out-of-sample error
    hatMatrix = regression.getHatMatrix()
    data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
    group = np.array(classify(data, slope, intercept)).reshape(-1, 1)
    predictedVals = hatMatrix.dot(group) 
    learnedGroup = [sign(predictedVals[i]) for i in range(sampleSize)]
    outSampleErr = sum([learnedGroup[i]!=group[i] for i in range(sampleSize)]) / float(sampleSize)

    print "out of sample error is:"
    print outSampleErr
    """ 

    # second experiment with regression result as initial value for perceptron
    experimentNum = 1000
    dataDim = 2
    lowerLim = -1
    upperLim = 1

    # experiment with sample size = 100
    sampleSize = 10
    iteration = 0
    for i in range(experimentNum):
        data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
        x1 = data[0, :]
        x2 = data[1, :]
        slope, intercept = findLine(x1, x2)
        group = classify(data, slope, intercept)
        print slope, intercept

        regression = Regression(data, group)
        regression.run()
        initCoeff = regression.getLinCoeff()
        print initCoeff
        sys.exit()
        perceptron = Perceptron(data, group)
        perceptron.setCoeff(initCoeff)
        perceptron.setMisclassified(regression.getMisclassifiedIndex())
        perceptron.learn()
        iteration += perceptron.getRepetition()
        
    print iteration/float(experimentNum) 


    
