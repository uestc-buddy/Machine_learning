#-*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

'''
import functions from other file
'''
import SVM_DataCreate as data_create

'''==============================================================================================
簡化SMO SVM算法
'''

'''
函數功能：在輸入的參數i和m之間隨機選擇一個不同於i的數字，也就是在選定了i之後隨機選取一個與之配對的alpha的取值的下標
'''
def selectJrand(i, m):
    j = i
    while j==i:
        j = int(np.random.uniform(0, m))

    return j

'''
函數功能：將輸入的元素限定在一個範圍內
'''
def clipAlpha(input, Low, high):
    if input>high:
        input = high
    if input<Low:
        input = Low

    return input

'''
簡化之後的SMO算法
dataMatin:      SVM訓練data輸入
classLabels:    SVM訓練data的分類標籤
C:              alpha值的界限C
toler:          容錯率
maxIter:        最大迭代次數
'''
def SmoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0   #分割直線的偏移量
    rows, cols = dataMatrix.shape
    alphas = np.mat(np.zeros((rows, 1))) #alpha matrix
    iter = 0    #iterator count
    while iter < maxIter:
        alphaPairChanged = 0
        for i in range(rows):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                j = selectJrand(i, rows) # SMO SVM簡化算法這裏使用隨機選取的方式進行選取與i配對的alpha值的下標
                fXj = float(np.multiply(alphas, labelMat[i]).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                '''公式(7)'''
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if H==L:
                    print "H==L program continued"
                    continue

                '''公式（8）（9）'''
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T -\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if 0<=eta:
                    print "eta>=0 program continued"
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], L, H)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print "j not moving enough %s" % ("program continued")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

                '''設置常數項 b '''
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i] and C>alphas[i]):
                    b = b1
                elif (0<alphas[j] and C>alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairChanged += 1
                print "iter: %d    i: %d, pairs changed: %d" % (iter, i, alphaPairChanged)

        if(alphaPairChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % (iter)

    return b, alphas

'''
函數功能：由計算出來的alphas獲得進行分類的權重向量
'''
def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels)
    rows, cols = np.shape(dataArr)
    w = np.mat(np.zeros((cols, 1)))
    for i in range(rows):
        w += np.multiply(alphas[i, :]*labelMat[:, i], X[i,:].T)
    return w

'''
main function
'''
Data, DataLabel = data_create.GenerateData(10)     #generate data
b, alphas = SmoSimple(Data, DataLabel, 2.0, 0.1, 50)

SuportPoint = []
for i in range(np.shape(alphas)[0]):
    if 0 < alphas[i]:
        SuportPoint.append([i, alphas[i]])
SuportPoint = np.mat(np.array(SuportPoint))
print SuportPoint
weight_vector = calcWs(alphas, np.mat(Data), np.mat(DataLabel))
print weight_vector

#fig, ax =plt.subplots(1,2,figsize=(10,5))
plt.figure("decsion boundry")
plt.title("decsion boundry")
for i in range(Data.shape[0]):
    if 1 == DataLabel[i]:
        plt.plot(Data[i][0], Data[i][1], 'r^')
    if -1 == DataLabel[i]:
        plt.plot(Data[i][0], Data[i][1], 'gs')

x = np.arange(0, 10, 0.5)
l = float( weight_vector[0, :])
k = np.multiply(x, l)
y = np.multiply(weight_vector[0, :], x) + weight_vector[1, :] + b
y = np.array(y)
plt.plot(x, y[0, :], 'b')
plt.xlabel("X label")
plt.ylabel("Y label")
plt.grid(True)
plt.title("input data distribution")
plt.show()

print b
print alphas