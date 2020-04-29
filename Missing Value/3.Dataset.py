import numpy as np
import csv

def dataLuar():
    data = []
    # ganti dataset buat perbandingan hasil handling missing value
    reader = open("remove_adult.data", "r")
    reader1 = reader.readlines()
    for i,row in enumerate(reader1):
        lines = row.split(",")
        word = 0
        if lines[14].replace("","").rstrip() == " <=50K":
            word = 1
        else:
            word = 2
        temp_line = [i+1, lines[0],lines[4].replace(" ",""),lines[10].replace(" ",""),lines[11].replace(" ",""),lines[12].replace(" ",""), word]
        data.append(temp_line)
    return  data


def simpanData(Data):
    np.savetxt('dataSet.csv', Data, delimiter=',', fmt='%s')
def keDataTrain(Data):
    np.savetxt('dataTrain.csv', Data[0:200], delimiter=',', fmt='%s')
def keDataTest(Data):
    Data = Data[200:400]
    for i,row in enumerate(Data):
        row[0] = i+1
        row.pop()
    np.savetxt('dataTest.csv', Data, delimiter=',', fmt='%s')
def keDataAsli(Data):
    Data = Data[200:400]
    for i,row in enumerate(Data):
        row[0] = i+1
    np.savetxt('dataAsli.csv', Data, delimiter=',', fmt='%s')
def keDataKelas1(Data):
    np.savetxt('dataKelas1.csv', Data[0:10000], delimiter=',', fmt='%s')
def keDataKelas2(Data):
    np.savetxt('dataKelas2.csv', Data[0:10000], delimiter=',', fmt='%s')
data = dataLuar()
keDataTrain(data)
keDataAsli(data)
keDataTest(data)