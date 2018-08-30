import numpy as np

def featureTransform(rawData):
    x1, x2, y = rawData[0], rawData[1], rawData[2]
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
    

dataFile = "in.dta"
feature = transformData(dataFile)



