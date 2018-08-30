import numpy as np
import sys

def sign(x):
    if np.sign(x)==0:
        return -1
    else:
        return int(np.sign(x))

def findLine(point1, point2):
    slope = (point2[1]-point1[1])/float(point2[0]-point1[0])
    intercept = point2[1] - slope*point2[0]
    return slope, intercept

def classify(data, slope, intercept):
    group = []
    size = np.size(data, 0)
    for i in range(size):
        group.append(sign(data[i,1]-slope*data[i,0]-intercept))
    return group

def findErrorRate(group1, group2):
    error = 0
    length = len(group1)
    for i in range(length):
        if group1[i] != group2[i]:
            error += 1
    return float(error)/length

class Perceptron(object): 
    def __init__(self, data, group):
        self.size = np.size(data, 0) 
        self.data = np.concatenate((np.ones((self.size,1)), data), axis=1) 
        self.correctGroup = group 
        self.misclassified = list(range(self.size)) 
        self.coeff = np.zeros((np.size(data, 1)+1, 1))
        self.classifiedGroup = [0] * self.size
        self.repeat = 0

    def setCoeff(self, coeff):
        self.coeff = coeff
        return self.coeff

    def setMisclassified(self, misclassified):
        self.misclassified = misclassified
        return self.misclassified

    def updateMisclassified(self):
        self.misclassified = []
        for i in range(self.size):
           if(self.classifiedGroup[i] != self.correctGroup[i]):
              self.misclassified.append(i) 
        return self.misclassified

    def recalculateClassified(self):
        for i in range(self.size):
            self.classifiedGroup[i] = sign(self.coeff.T.dot(self.data[i, :].reshape(-1, 1)))
        return self.classifiedGroup

    def updateCoeff(self):
        if self.misclassified:
            index = self.misclassified[0]
            self.coeff = self.coeff + (self.correctGroup[index]*self.data[index, :]).reshape(-1, 1)
    
    def learn(self):
        while self.misclassified != []:
            self.repeat += 1
            self.updateCoeff()
            self.recalculateClassified()
            self.updateMisclassified()
        return self.repeat

    def getCoeff(self):
        return self.coeff

    def getRepetition(self):
        return self.repeat

    



if __name__ == "__main__":
    experimentNum = 1000
    dataDim = 2
    lowerLim = -1
    upperLim = 1


    # experiment with sample size = 10
    sampleSize = 10
    repetition = []
    
    for i in range(experimentNum):
        data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
        x1 = data[0, :]
        x2 = data[1, :]
        slope, intercept = findLine(x1, x2)
        group = classify(data, slope, intercept)

        perceptron = Perceptron(data, group)
        perceptron.learn()
        repetition.append(perceptron.getRepetition())

    repetition.sort()
    print "For sample size = 10, the average repetition is:"
    print sum(repetition)/experimentNum
    print repetition[len(repetition)/2]

    weight = perceptron.getCoeff()
    learnedSlope = -float(weight[1])/weight[2]
    learnedInter = -float(weight[0])/weight[2]

    testData = np.random.uniform(lowerLim, upperLim, dataDim*experimentNum).reshape(experimentNum, -1)
    testGroupClass = classify(testData, slope, intercept)
    learnedClass = classify(testData, learnedSlope, learnedInter) 
    errorRate = findErrorRate(testGroupClass, learnedClass)
    print "And the error rate is:"
    print errorRate

    
    # experiment with sample size = 100
    sampleSize = 100
    repetition = []

    for i in range(experimentNum):
        data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
        x1 = data[0, :]
        x2 = data[1, :]
        slope, intercept = findLine(x1, x2)
        group = classify(data, slope, intercept)

        perceptron = Perceptron(data, group)
        perceptron.learn()
        repetition.append(perceptron.getRepetition())

    repetition.sort()
    print "For sample size = 100, the average repetition is:"
    print sum(repetition)/experimentNum
    print repetition[len(repetition)/2]

    weight = perceptron.getCoeff()
    learnedSlope = -float(weight[1])/weight[2]
    learnedInter = -float(weight[0])/weight[2]

    testData = np.random.uniform(lowerLim, upperLim, dataDim*experimentNum).reshape(experimentNum, -1)
    testGroupClass = classify(testData, slope, intercept)
    learnedClass = classify(testData, learnedSlope, learnedInter) 
    errorRate = findErrorRate(testGroupClass, learnedClass)
    print "And the error rate is:"
    print errorRate

