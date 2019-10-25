# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:28:52 2019

@author: Chandramauli Pandya
"""

# Required import statements
import numpy as np
import random


def readData(fileName):
    #Read input data file
    #df = open(fileName)
    dataFileOld = open(fileName)
    li = dataFileOld.readlines()
    dataFileOld.close()
    #Shuffle data file
    random.shuffle(li)
    
    fid = open("shuffled_data.txt", "w")
    fid.writelines(li)
    fid.close()
    fid = open("shuffled_data.txt")
    
    return fid

def processData(fileObj):
    dataMatrix = []
    labelMatrix = []
    for a in fileObj:
        dataArray=a.split(",")
        
        if(len(dataArray)==5):
            b = []
            b.append(float(1))
            
            sepalLength = dataArray[0]
            b.append(float(sepalLength))
            sepalWidth = dataArray[1]
            b.append(float(sepalWidth))
            petalLength = dataArray[2]
            b.append(float(petalLength))
            petalWidth = dataArray[3]
            b.append(float(petalWidth))
            
            classType = dataArray[4].split("\n")[0]
            
            dataMatrix.append(b)
            if(classType=='Iris-setosa'):
                labelMatrix.append(1)
            elif(classType=='Iris-versicolor'):
                labelMatrix.append(2)
            else:
                labelMatrix.append(3)
        
        npData = np.asarray(dataMatrix)
        
    return npData, labelMatrix

# Function to find model parameter matrix
def findParams(npData, labelMatrix):
    transData = npData.transpose()
    FinalParam1 = np.matmul(transData, npData)
    
    FinalParam2 = np.linalg.pinv(FinalParam1)
    
    FinalParam3 = np.matmul(FinalParam2, transData)
    
    FinalParam = np.matmul(FinalParam3, labelMatrix)
    
    return FinalParam
  
# Function to predict values using model parameter matrix
def predictValues(data, FinalParam):
    return np.matmul(data, FinalParam)

# Function to find accuracy
def find_accuracy(l1,l2):
    
    count=0
    count2=0
    for k in l1:
        if(k==l2[count]):
            count2+=1
        count+=1
    
    percentage = count2*100/len(l1)
    
    return percentage
    
# Function to split data for training and testing using K-fold method
def k_fold(XData, YData, k):
    splitXData = np.vsplit(XData, k)
   
    accuracyList = []
    paramList = []
    splitYData = np.array_split(YData, k)
   
    count=0
    
    for a in splitYData:
        
        dataMat=[]
        labelMat=[]
        for p in range(len(splitXData)):
        
            if(p!=count):
                for o in splitXData[p]:
                    dataMat.append(o)
        #print(len(dataMat))
        for p in range(len(splitYData)):
          
           if(p!=count):
               for r in splitYData[p]:
                    labelMat.append(r)
        #print(labelMat)
        
        param = findParams(np.asarray(dataMat),np.asarray(labelMat))
        
        op = predictValues(splitXData[count],param)
        predicted = np.rint(op).astype(int)
            
        accuracy = find_accuracy(splitYData[count],predicted)
        
        accuracyList.append(accuracy)
        paramList.append(param)
        count+=1
    
    paramSum=np.zeros(5)
    for i in paramList:
        #print(i)
        paramSum=np.add(paramSum, i)
        #print(i)
    #print(paramSum)
    paramListAvg=[]
    for j in paramSum:
        paramListAvg.append(j/10)
        
    averageValues = np.average(accuracyList)
    
    print("Accuracy of the model when k is {}:".format(k))
    print(round(averageValues,2))



if __name__ == '__main__':
  
    # Function calling
    fileObj = readData("iris.data")
    npData, labelMatrix = processData(fileObj)

    print("-----------------------------------")
    k_fold(npData, labelMatrix, 10)
    k_fold(npData, labelMatrix, 3)
    k_fold(npData, labelMatrix, 5)
    print("-----------------------------------")
    print("Note: Accuracy will be very on every run as the data file will shuffle randomly every time.")