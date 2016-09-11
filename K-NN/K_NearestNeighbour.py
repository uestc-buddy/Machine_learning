#coding = utf-8
import numpy as np
import matplotlib.pyplot as plt



#read input data
#=============================================================================
def LoadData(filename, split_str = '\t'):
    data_file = open(filename, "r")
    arrayOflines = data_file.readlines()    #get string array
    rowOflines = len(arrayOflines)          #get arrayOflines row number
    featureData = np.zeros(rowOflines, 3)
    classlabelVector = []                   #record the feture's ID
    count = 0
    data_file.close()                       #close file
    for item in arrayOflines:
        item = item.strip()                 #delete space infront and end
        item_get = item.split(split_str)    #split the item string
        featureData[count, :] = item_get[0:3]
        count += 1                          #increase count unmber
        classlabelVector.append(item_get[-1])
    return featureData, classlabelVector


#=============================================================================
def K_Classify(featureData, classlabelVector, test_data, K_number):
    fetureRows = featureData.shape[0]           #get the training data row number
    diffMat = featureData - test_data
    sqDiffMat = diffMat**2
    distance = sqDiffMat.sum(axis=1)    #the sum of row data
    distance = distance**0.5
    sortIndex = np.argsort(distance)

    lable = np.zeros(K_number)
    for i in range(K_number):
        if 1 == classlabelVector[sortIndex[i]]:
            lable[0] += 1
        if 2 == classlabelVector[sortIndex[i]]:
            lable[1] += 1
        if 3 == classlabelVector[sortIndex[i]]:
            lable[2] += 1
    print lable
    result = np.argsort(lable)
    print result
    print "test_data: " + test_data.__str__() + " is belong to class: " + (result[4]+1).__str__()
#    print distance
#
# print sortIndex

#=============================================================================
def GnerateData(DataNumber=10, class_num = 3):
    data = np.zeros((DataNumber*class_num, class_num))

    for i in range(1, class_num+1):
        for row_data in range(0, DataNumber):
            line = np.random.uniform(1, 10, 2).astype(int)
            line = np.hstack((line, i))
            #print line
            data[row_data+(i-1)*DataNumber, :] = line
            #data = np.vstack((data, line))

    return data, DataNumber, class_num

#=============================================================================
def PrintInfro(about_infor, data):
    print "\n"
    print about_infor
    print data
    print "\n"

#=============================================================================
def PlotData(train_data, DataNumberPerClass, test_data, class_num=3):
    #index_1 = np.argwhere(train_data[:, 2] == 1)
    #index_2 = np.argwhere(train_data[:, 2] == 2)
    #index_3 = np.argwhere(train_data[:, 2] == 3)

    class1_data = train_data[0:DataNumberPerClass, 0:2]
    class2_data = train_data[DataNumberPerClass:DataNumberPerClass*2, 0:2]
    class3_data = train_data[DataNumberPerClass*2:, 0:2]
    plt.figure("DATA")
    plt.plot(class1_data[:, 0], class1_data[:, 1], "b^", class2_data[:, 0], class2_data[:, 1], "g.",
             class3_data[:, 0], class3_data[:, 1], "ys")
    plt.hold(True)
    plt.plot(test_data[0], test_data[1], "rs")
    plt.xlabel("X1")
    label_range = [-2, 12, -2, 12]
    plt.axis(label_range)
    plt.ylabel("X2")
    plt.title("Feture Data")
    plt.show()

# thr main function
#=============================================================================
filestring = "train_data.txt"
train_data, DataNumberPerClass, class_num = GnerateData(20, 3)
PrintInfro("trian-data: ", train_data)
PrintInfro("DataNumberPerClass: ", DataNumberPerClass)
PrintInfro("Class_num: ", class_num)

test_data = np.random.randint(1, 10, 2)
PlotData(train_data, DataNumberPerClass, test_data)
classlabelVector = train_data[:, 2]
train_data = train_data[:, 0:2]
K_Classify(train_data, classlabelVector, test_data, 5)
#featureData, classlabelVector = LoadData(filestring)
