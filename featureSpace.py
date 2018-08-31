import numpy as np
import sys
from regression import *

def featureTransform(rawData):
    x1, x2, y = rawData[0], rawData[1], int(rawData[2])
    feature = [x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2), y]
    return feature

def getFeature(rawInput) :
    feature = map(featureTransform, rawInput)
    return feature 

def transformData(fileName):
    dataFile = open(fileName)
    rawData = [map(eval, line.strip(" ").rstrip("\r\n").split("  ")) for line in dataFile]
    feature = np.array(getFeature(rawData))
    return feature
    

trainDataFile = "in.dta"
trainData = transformData(trainDataFile)
trainData, trainGroup = trainData[:, :-1], trainData[:, -1]
testDataFile = "out.dta"
testData = transformData(testDataFile)
testData, testGroup = testData[:, :-1], testData[:, -1]

# no regularization
regression = Regression(trainData, trainGroup, testData, testGroup, 0)
regression.run()
print regression.getInSampleErr()
print regression.getOutSampleErr()

# penalty lambda = 10^(-3)
regression = Regression(trainData, trainGroup, testData, testGroup, 10**(-3))
regression.run()
print regression.getInSampleErr()
print regression.getOutSampleErr()

# regularization parameter lambda = 10^3
regression = Regression(trainData, trainGroup, testData, testGroup, 10**3)
regression.run()
print regression.getInSampleErr()
print regression.getOutSampleErr()

for i in range(-2, 3):
    regression = Regression(trainData, trainGroup, testData, testGroup, 10**i)
    regression.run()
    print "When regularization parameter = ", i
    print regression.getInSampleErr()
    print regression.getOutSampleErr()
    
