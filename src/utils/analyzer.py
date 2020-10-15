#!/usr/bin/python

import pandas as pd
import math
import numpy

class Analyzer:

    def __init__(self,listKeyClasses):
        self.__listKeysClasses = listKeyClasses
        self.__createConfusionMatrix(listKeyClasses)

    def calcAccuracy(self):
        sumTrues = 0.0
        sumAll = 0.0
        for keyPredicted in self.__listKeysClasses:
            sumTrues += self.__confusionMatrix[keyPredicted].loc[keyPredicted]
            for keyOriginal in self.__listKeysClasses:
                    sumAll += self.__confusionMatrix[keyPredicted].loc[keyOriginal]

        return sumTrues/sumAll

    def calcRecall(self, precType = "micro"):
        recall = 0.0

        if precType == "macro":
            for key in self.__listKeysClasses:
                sumClass = 0.0
                for keyPredicted in self.__listKeysClasses:
                    # sum = VP + FN
                    sumClass += self.__confusionMatrix[keyPredicted].loc[key]
                # prec = VP / sum
                recall += self.__confusionMatrix[key].loc[key] / sumClass
            # precision Macro
            recall = recall/len(self.__listKeysClasses)
        elif precType == "micro":
                sumClass = 0.0
                for key in self.__listKeysClasses:
                    for keyPredicted in self.__listKeysClasses:
                        #sum = VP1 + FN1 + ... + VPn + FNn
                        sumClass += self.__confusionMatrix[keyPredicted].loc[key]
                    # prec = VP1 + ... + VPn
                    recall += self.__confusionMatrix[key].loc[key]
                #precison Micro
                recall = recall/sumClass

        return recall

    def calcPrecision(self, precType = "micro"):
        prec = 0.0

        if precType == "macro":
            for key in self.__listKeysClasses:
                sumClass = 0.0
                for keyOriginal in self.__listKeysClasses:
                    #sum = VP + FP
                    sumClass += self.__confusionMatrix[key].loc[keyOriginal]
                # prec = VP / sum
                prec += self.__confusionMatrix[key].loc[key] / sumClass
            # precision Macro
            prec = prec/len(self.__listKeysClasses)
        elif precType == "micro":
                sumClass = 0.0
                for key in self.__listKeysClasses:
                    for keyOriginal in self.__listKeysClasses:
                        #sum = VP1 + FP1 + ... + VPn + FPn
                        sumClass += self.__confusionMatrix[key].loc[keyOriginal]
                    # prec = VP1 + ... + VPn
                    prec += self.__confusionMatrix[key].loc[key]
                #precison Micro
                prec = prec/sumClass
                
        return prec

    def calcFBethaMeasure(self, betha,precType = "micro"):
        num = self.calcPrecision(precType) * self.calcRecall(precType)
        denom = (pow(betha,2)*self.calcPrecision(precType) + self.calcRecall(precType))
        return (1+pow(betha,2))*((num)/(denom))

    def __createConfusionMatrix(self,listKeyClasses):
        self.__confusionMatrix = pd.DataFrame(0.0,index=listKeyClasses,columns=listKeyClasses)
        
    def addValueInConfusionMatrix(self,keyClassPredicted, keyClassOriginal):
        if(self.__confusionMatrix[keyClassPredicted].loc[keyClassOriginal] > 0.0):
            self.__confusionMatrix[keyClassPredicted].loc[keyClassOriginal] += 1.0
        else:
            self.__confusionMatrix[keyClassPredicted].loc[keyClassOriginal] = 1.0

    def getConfusionMatrix(self):
        return self.__confusionMatrix

    @classmethod
    def calcAverage(cls,valuesList):
        return numpy.mean(valuesList)

    @classmethod
    def calcStandarDeviation(cls,valuesList):
        return numpy.std(valuesList)

    @classmethod
    def calcPercentile(cls,valuesList, percent):
        return numpy.percentile(valuesList, percent)

    @classmethod
    def calcMedian(cls,valuesList):
        return numpy.median(valuesList)

    @classmethod
    def getPercentOfClasses(cls,data,column):
        auxPercent = {}
        countQtdOfInstances = 0
        percentValuesByClass = {}

        for cellValue in data[column]:
            countQtdOfInstances += 1
            if(cellValue in auxPercent):
                auxPercent[cellValue] += 1
            else:
                auxPercent[cellValue] = 1
        
        for key in auxPercent.keys():
            percentValuesByClass[key] = (auxPercent[key]/countQtdOfInstances)*100.0

        return percentValuesByClass