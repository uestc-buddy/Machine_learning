#-*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def GenerateData(nums):

    '''
    x = []
    for i in range(0, nums, 1):
        x.append(i)
    '''
    x = np.linspace(0, 30, nums)
    y = DataFounction(x)

    DataCreated = np.zeros((nums, 3))
    DataCreated[:, 0] = np.array(x)
    DataCreated[:, 1] = 1
    DataCreated[:, 2] = np.array(y)
    #print DataCreated
    ShowData(DataCreated)
    #print DataLabel
    return DataCreated

def DataFounction(inputX):
    inputX = np.mat(inputX)
    y = 0.8*inputX - 5 + 2*np.sin(inputX) - 10*np.random.random_sample((1,1))
    return y

def ShowData(Data, Datalabel = None, IsShow = True):
    Data_rows = Data.shape[0]
    if None != Datalabel:
        Datalabel_rows = Datalabel.shape[0]
        if Data_rows != Datalabel_rows:
            print "ShowData: Data_rows!=Datalabel_rows"
            return
        if IsShow:
            plt.figure("the input data")
            for i in range(Data_rows):
                if 1 == Datalabel[i]:
                    plt.plot(Data[i][0], Data[i][1], 'r^')
                if -1 == Datalabel[i]:
                    plt.plot(Data[i][0], Data[i][1], 'gs')
            #plt.ion()
            plt.xlabel("X label")
            plt.ylabel("Y label")
            plt.grid(True)
            plt.title("input data distribution")
            plt.show()
        else:
            print "Show Data has been denied"
    elif None == Datalabel:
        plt.figure("the input Data")
        plt.plot(Data[:, 0], Data[:, 2], 'g.')
        plt.xlabel("X label")
        plt.ylabel("y label")
        plt.title("input data distributon")
        plt.show()


def SaveData2Txt(Data, Datalabel, filepath = "Data.txt"):
    if "Data.txt" == filepath:
        print "there is no input file-path name, will generate in locally\n"
    try:
        myfile = open(filepath, 'w')
        rows = Data.shape[0]
        for i in range(rows):
            myfile.writelines(Data[i][0].__str__() + "\t" + Data[i][1].__str__() + "\t" +
                              Datalabel[i].__str__() + "\n")
        myfile.close()
    except IOError, e:
        print "write data to %s failed!" % filepath

def ReadData2Mat(filepath = "Data.txt"):
    if "Data.txt" == filepath:
        print "will open %s as Data input\n" % (filepath)
    Data = []
    Label = []
    try:
        myfile = open(filepath, 'r')
        count = 0
        for line in myfile.readlines():
            lineArr = line.strip().split('\t')
            Data.append([float(lineArr[0]), float(lineArr[1])])
            Label.append([float(lineArr[2])])
            count += 1
        myfile.close()
    except IOError, e:
        print "open and read %s error\n" % (filepath)

    Data = np.mat(Data)
    Label = np.mat(Label)
    return Data, Label


#the test main founction
#data = GenerateData(50)

