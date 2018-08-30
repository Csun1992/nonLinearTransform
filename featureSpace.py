import numpy as np
import sys

def featureTransform(rawData):
    x1, x2, y = rawData[0], rawData[1], int(rawData[2])
    feature = [1, x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2), y]
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
testDataFile = "out.dta"
testData = transformData(testDataFile)





