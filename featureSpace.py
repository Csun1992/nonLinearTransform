import numpy as np

def featureTransform(rawData):
    x1, x2 = rawData[0], rawData[1]
    feature = [1, x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2)]
    return feature

def getFeature(rawInput) :
    feature = map(featureTransform, rawInput)
    return feature 

dataFile = open('in.dta')
rawData = [map(eval, line.strip(" ").rstrip("\r\n").split("  ")) for line in dataFile]
feature = np.array(getFeature(rawData))



