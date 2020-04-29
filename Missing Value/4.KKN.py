import csv
import numpy as np
import math
from decimal import Decimal
import time

class dataTrain:
    def __init__(self, index = None, X1 = None, X2 = None, X3 = None, X4 = None, X5 = None, Y = None):
        self.index = index
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.X5 = X5
        self.Y = Y

class dataTest:
    def __init__(self, index = None, X1 = None, X2 = None, X3 = None, X4 = None, X5 = None):
        self.index = index
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.X5 = X5
        self.Y = ""
        self.Neighbour = ""

def simpanData(listIndex, listX1, listX2, listX3, listX4, listX5, listY, ListNeighbour):
    np.savetxt('dataTest_Hasil.csv', [p for p in zip(listIndex, listX1, listX2, listX3, listX4, listX5, listY, ListNeighbour)], delimiter=',', fmt='%s')

def rumus(X1t, X2t, X3t, X4t, X5t, X1tr, X2tr, X3tr, X4tr, X5tr):
    return math.sqrt((X1tr-X1t)**2+(X2tr-X2t)**2+(X3tr-X3t)**2+(X4tr-X4t)**2+(X5tr-X5t)**2)

def dataLuar():
    data = []
    reader = csv.reader(open("dataTrain.csv"), delimiter=",")
    for row in reader:
        data.append(dataTrain(row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
    return data

def dataLuar2():
    data = []
    reader = csv.reader(open("dataTest.csv"), delimiter=",")
    for row in reader:
        data.append(dataTrain(row[0], row[1], row[2], row[3], row[4], row[5]))
    return data

def dataLuar3():
    data = []
    reader = csv.reader(open("dataAsli.csv"), delimiter=",")
    for row in reader:
        data.append(row[6])
    return data


def PerulanganK(dataTrainn, dataTestt, K):
    i = 1
    for row in dataTestt:
        ListAll = []
        for row2 in dataTrainn:
            ListAll.append([rumus(float(row.X1), float(row.X2), float(row.X3), float(row.X4), float(row.X5), float(row2.X1), float(row2.X2), float(row2.X3), float(row2.X4), float(row2.X5)), row2.Y, row2.index])
        ListAll.sort()
        ListRangeK = []
        ListAlamat = []
        for x in range (0, K):
            ListRangeK.extend(ListAll[x][1])
            ListAlamat.append([ListAll[x][1], ListAll[x][2]])
        dataTestt[int(row.index) - 1].Y = max(set(ListRangeK), key=ListRangeK.count)
        Class = dataTestt[int(row.index) - 1].Y
        ListAlamat = [x for x in ListAlamat if x[0] == Class]
        dataTestt[int(row.index) - 1].Neighbour = [x[1] for x in ListAlamat]
        print(i, ". Kelas = ",dataTestt[int(row.index) - 1].Y, "Karena bertetangga dengan ", dataTestt[int(row.index) - 1].Neighbour)
        i += 1
    return dataTestt

def Akurasi(DataAsli, DataPrediksi):
    i = 0
    for x in range(len(DataAsli)):
        if DataAsli[x] == DataPrediksi[x]:
            i += 1
    akurasi = i / len(DataAsli) * 100
    return akurasi


start = time.time()
dataTrainn = dataLuar() #data train
dataTestt = dataLuar2() #data test
listIndex = ["Index"]
listX1 = ["X1"]
listX2 = ["X2"]
listX3 = ["X3"]
listX4 = ["X4"]
listX5 = ["X5"]
listY = ["Y"]
listNeighbour = ["Neighbour"]
K = 5
PerulanganK(dataTrainn, dataTestt, K)
end = time.time()
print("Index | X1 | X2 | X3 | X4 | X5 | Y")
for row in dataTestt:
    listIndex.append(row.index)
    listX1.append(row.X1)
    listX2.append(row.X2)
    listX3.append(row.X3)
    listX4.append(row.X4)
    listX5.append(row.X5)
    listY.append(row.Y)
    listNeighbour.append(row.Neighbour)
    print(row.index, "|", row.X1, "|", row.X2, "|", row.X3, "|", row.X4, "|", row.X5, "|", row.Y, "|", row.Neighbour)
dataAsli = dataLuar3()
akurasi = Akurasi(dataAsli, listY[1:])
print("\nAkurasi sebesar", akurasi, "%")
listIndex.append(len(listIndex)+1)
listX1.append("Akurasi")
listX2.append(akurasi)
listX3.append("%")
listX4.append("")
listX5.append("")
listY.append("")
listNeighbour.append("")
simpanData(listIndex, listX1, listX2, listX3, listX4, listX5, listY, listNeighbour)
print("\nHasil Juga Disimpan Dalam File dataTest_Hasil.CSV")
print("Hasil Running Selama = ", end-start, "Detik")
