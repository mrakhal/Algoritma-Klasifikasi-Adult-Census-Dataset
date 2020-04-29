from pandas import read_csv, DataFrame
import pandas as pd
import sys
from pandas_profiling import ProfileReport

# Fungsi untuk membaca Dataset
def bacaDataset(namafile):
    df = pd.read_csv(namafile)
    return df

# Strategi 1 : Ganti semua NaN dengan Value yang paling sering muncul pada suatu atribut (cek output.html)
def replace_missing(data_set):
	data_set['WORKCLASS'] = data_set['WORKCLASS'].replace(to_replace = ' ?', value = ' Private')
	data_set['EDUCATION'] = data_set['EDUCATION'].replace(to_replace = ' ?', value = ' HS-grad')
	data_set['MARITAL STATUS'] = data_set['MARITAL STATUS'].replace(to_replace = ' ?', value = ' Married-civ-spouse')
	data_set['OCCUPATION'] = data_set['OCCUPATION'].replace(to_replace = ' ?', value = ' Prof-specialty')
	data_set['RELATIONSHIP'] = data_set['RELATIONSHIP'].replace(to_replace = ' ?', value = ' Husband')
	data_set['RACE'] = data_set['RACE'].replace(to_replace = ' ?', value = ' White')
	data_set['SEX'] = data_set['SEX'].replace(to_replace = ' ?', value = ' Male')
	data_set['NATIVE-COUNTRY'] = data_set['NATIVE-COUNTRY'].replace(to_replace = ' ?', value = ' United-States')

	data_set = data_set.to_csv('replace_adult.data', sep =',', index = False)	#Output file 1
	return data_set

# Strategi 2 : Hapus Record yang memiliki missing value
def remove_missing(data_set):
	
	data_set = data_set[data_set['WORKCLASS'] != ' ?']
	data_set = data_set[data_set['EDUCATION'] != ' ?']
	data_set = data_set[data_set['MARITAL STATUS'] != ' ?']
	data_set = data_set[data_set['OCCUPATION'] != ' ?']
	data_set = data_set[data_set['RELATIONSHIP'] != ' ?']
	data_set = data_set[data_set['RACE'] != ' ?']
	data_set = data_set[data_set['SEX'] != ' ?']
	data_set = data_set[data_set['NATIVE-COUNTRY'] != ' ?']
	
	data_set = data_set.to_csv('remove_adult.data', sep = ',', index = False)	#Output file 2
	return data_set

# Fungsi Main
def main():
    namafile = ("adult.data")
    df = bacaDataset(namafile)
    df.columns = ['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION-NUM','MARITAL STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL-GAIN','CAPITAL-LOSS','HOURS-PER-WEEK','NATIVE-COUNTRY','CLASS']
   
    remove_missing(df)
    replace_missing(df)

main()  



