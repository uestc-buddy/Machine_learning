#-*- coding: UTF-8 -*-
import DataCreate
import numpy as np
import matplotlib.pyplot as plt

'''================================================================================='''
#普通二次回歸
def GetWMatrix(XData, YData):
    if None==XData or None==YData:
        return None
    xTx = XData.T * XData
    if 0 == np.linalg.det(xTx):
        print "this matrix cannot do inverse"
        return None
    W = xTx.I * XData.T * YData
    return W

'''================================================================================='''
def calcY(TestData, XDataIn, YDatain, theta=0.5):
    XDataIn = np.mat(XDataIn)
    YDatain = np.mat(YDatain)
    rows, cols = np.shape(XDataIn)
    weight = np.mat(np.eye(rows))
    for i in range(0, rows, 1):
        diff = TestData - XDataIn[i, :]
        weight[i, i] = np.exp(diff * diff.T / (-2.0*theta**2))
    XTX = XDataIn.T * weight * XDataIn
    if 0 == np.linalg.det(XTX):
        print "this matrix cannot do inverse"
        return None
    w = XTX.I * XDataIn.T * weight * YDatain
    value = TestData * w
    return value

#加權線性迴歸
def GetY(TestData, XDataIn, YDatain, theta=0.5):
    TestData = np.mat(TestData)
    rows, cols = np.shape(XDataIn)
    result = np.zeros(rows)
    for i in range(0, rows, 1):
        result[i] = calcY(TestData[i, :], XDataIn, YDatain, theta)
    return result


'''================================================================================='''
#main
nums = 100
data = DataCreate.GenerateData(nums)
XData = np.mat(data[:, 0:-1])
YData = np.mat(data[:, -1]).T
W1 = GetWMatrix(XData, YData)
if None == W1:
    print "None == W1"
print W1


plt.figure("the result plot")
x = np.linspace(0, 30, nums)
x = np.hstack((np.mat(x).T, np.ones((nums, 1))))
print x

Y = GetY(x, XData, YData, 1)
print Y

#plt.plot(XData[:, 0], YData, 'b.', XData[:, 0], Y, 'b*')
plt.plot(XData[:, 0], YData, 'b.', XData[:, 0], XData*W1, 'r-', XData[:, 0], Y, 'b*')
plt.show()

