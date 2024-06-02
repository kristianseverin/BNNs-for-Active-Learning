import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dtw import *

# Load the data
ALL3 = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/AII 3.0_final.csv', sep=';')
SCL92 = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/SCL-92_final.csv', sep=';')
BDI = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/BDI-II_final.csv', sep=';')
BERN = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/Bern_final.csv', sep=';')
IIP64 = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/IIP64_final.csv', sep=';')
RFQ = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/RFQ_final.csv', sep=';')
SIPP = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/SIPP-SV_final.csv', sep=';')
TPQ = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/TPQ_final.csv', sep=';')
WAI_patient = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/WAI_patient_final.csv', sep=';')
WAI_therapist = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/Final Processed Questionnaire Files/WAI_therapist_final.csv', sep=';')

dataframes = [ALL3, SCL92, BDI, BERN, IIP64, RFQ, SIPP, TPQ, WAI_patient] # WAI_therapist is not included as it relates to the therapist and not the patient

# Rename all columns called 'Patient ID' to 'Patient'
for i in range(len(dataframes)):
    df = dataframes[i]
    if 'Patient ID' in df.columns:
        df.rename(columns={'Patient ID': 'Patient'}, inplace=True)
    if 'PATIENT' in df.columns:
        df.rename(columns={'PATIENT': 'Patient'}, inplace=True)

# This is the IDs I have permission to do analyses on
allowed_ids = ['ACL21', 'CK23', 'HHC07', 'HSN28', 'JLIHN28', 'KTO2', 'LSM13', 'MGH04', 'MP15', 'NMJ10', 'NSP24']

# For all the dataframes only keep these ID's
for i in range(len(dataframes)):
    df = dataframes[i]
    df = df[df['Patient'].isin(allowed_ids)]
    dataframes[i] = df

# Unlist the dataframes and assign back to their original names
ALL3, SCL92, BDI, BERN, IIP64, RFQ, SIPP, TPQ, WAI_patient = dataframes


with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S11_20170518_P_RR.txt') as f: 
    MP15_S11_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S11_20170518_T_RR.txt') as f:
    MP15_S11_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S12_20170530_P_RR.txt') as f: 
    MP15_S12_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S12_20170530_T_RR.txt') as f:
    MP15_S12_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S13_20170606_P_RR.txt') as f: 
    MP15_S13_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S13_20170606_T_RR.txt') as f:
    MP15_S13_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S14_20170613_P_RR.txt') as f: 
    MP15_S14_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S14_20170613_T_RR.txt') as f:
    MP15_S14_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S15_20170622_P_RR.txt') as f:
    MP15_S15_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S15_20170622_T_RR.txt') as f:
    MP15_S15_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S16_20170629_P_RR.txt') as f:
    MP15_S16_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S16_20170629_T_RR.txt') as f:
    MP15_S16_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S18_20170810_P_RR.txt') as f:
    MP15_S18_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S18_20170810_T_RR.txt') as f:
    MP15_S18_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S19_20170824_P_RR.txt') as f:
    MP15_S19_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S19_20170824_T_RR.txt') as f:
    MP15_S19_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S20_20170928_P_RR.txt') as f:
    MP15_S20_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S20_20170928_T_RR.txt') as f:
    MP15_S20_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S21_20171010_P_RR.txt') as f:
    MP15_S21_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S21_20171010_T_RR.txt') as f:
    MP15_S21_T = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S22_20171019_P_RR.txt') as f:
    MP15_S22_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/MP15/MP15_S22_20171019_T_RR.txt') as f:
    MP15_S22_T = f.readlines()  
# HHC07
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/HHC07/HHC07_S06_20190403_P_RR.txt') as f: 
    HHC07_S06_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/HHC07/HHC07_S06_20190403_T_RR.txt') as f:
    HHCO7_S06_T = f.readlines()
# LSM13
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/LSM13/LSM13_S03_20170919_P_RR.txt') as f: 
    LSM13_S03_P = f.readlines()
with open('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HRV-RR/LSM13/LSM13_S03_20170919_T_RR.txt') as f:
    LSM13_S03_T = f.readlines()

# function to preprocess the data
def preprocessTimeseries(series1, series2):
    df_p = pd.DataFrame(series1)
    df_t = pd.DataFrame(series2)
    df = pd.concat([df_p, df_t], axis=1)
    df.columns = ['Patient', 'Therapist']
    df = df.iloc[7:]
    df = df.dropna()
    df['Patient'] = df['Patient'].str.replace(',\n', '')
    df['Therapist'] = df['Therapist'].str.replace(',\n', '')
    df['Patient'] = pd.to_numeric(df['Patient'])
    df['Therapist'] = pd.to_numeric(df['Therapist'])
    return df

# preprocess all dyads
df_MP15_S11 = preprocessTimeseries(MP15_S11_P, MP15_S11_T)
df_MP15_S12 = preprocessTimeseries(MP15_S12_P, MP15_S12_T)
df_MP15_S13 = preprocessTimeseries(MP15_S13_P, MP15_S13_T)
df_MP15_S14 = preprocessTimeseries(MP15_S14_P, MP15_S14_T)
df_MP15_S15 = preprocessTimeseries(MP15_S15_P, MP15_S15_T)
df_MP15_S16 = preprocessTimeseries(MP15_S16_P, MP15_S16_T)
df_MP15_S18 = preprocessTimeseries(MP15_S18_P, MP15_S18_T)
df_MP15_S19 = preprocessTimeseries(MP15_S19_P, MP15_S19_T)
df_MP15_S20 = preprocessTimeseries(MP15_S20_P, MP15_S20_T)
df_MP15_S21 = preprocessTimeseries(MP15_S21_P, MP15_S21_T)
df_MP15_S22 = preprocessTimeseries(MP15_S22_P, MP15_S22_T)
df_HHC07_S06 = preprocessTimeseries(HHC07_S06_P, HHCO7_S06_T)
df_LSM13_S03 = preprocessTimeseries(LSM13_S03_P, LSM13_S03_T)

# function to calculate the DTW distance
def calculateDTW(series1, series2):
    distance = dtw(series1, series2, keep_internals=True)
    return distance

# calculate the DTW distance for all dyads
print('calculating DTW for MP15_S11')
#dtw_MP15_S11 = calculateDTW(df_MP15_S11['Patient'], df_MP15_S11['Therapist'])
print('calculating DTW for MP15_S12')
#dtw_MP15_S12 = calculateDTW(df_MP15_S12['Patient'], df_MP15_S12['Therapist'])
print('calculating DTW for MP15_S13')
#dtw_MP15_S13 = calculateDTW(df_MP15_S13['Patient'], df_MP15_S13['Therapist'])
print('calculating DTW for MP15_S14')
#dtw_MP15_S14 = calculateDTW(df_MP15_S14['Patient'], df_MP15_S14['Therapist'])
print('calculating DTW for MP15_S15')
#dtw_MP15_S15 = calculateDTW(df_MP15_S15['Patient'], df_MP15_S15['Therapist'])
print('calculating DTW for MP15_S16')
#dtw_MP15_S16 = calculateDTW(df_MP15_S16['Patient'], df_MP15_S16['Therapist'])
print('calculating DTW for MP15_S18')
#dtw_MP15_S18 = calculateDTW(df_MP15_S18['Patient'], df_MP15_S18['Therapist'])
print('calculating DTW for MP15_S19')
#dtw_MP15_S19 = calculateDTW(df_MP15_S19['Patient'], df_MP15_S19['Therapist'])
print('calculating DTW for MP15_S20')
#dtw_MP15_S20 = calculateDTW(df_MP15_S20['Patient'], df_MP15_S20['Therapist'])
print('calculating DTW for MP15_S21')
#dtw_MP15_S21 = calculateDTW(df_MP15_S21['Patient'], df_MP15_S21['Therapist'])
print('calculating DTW for MP15_S22')
dtw_MP15_S22 = calculateDTW(df_MP15_S22['Patient'], df_MP15_S22['Therapist'])
print('calculating DTW for HHC07_S06')
dtw_HHC07_S06 = calculateDTW(df_HHC07_S06['Patient'], df_HHC07_S06['Therapist'])
print('calculating DTW for LSM13_S03')
dtw_LSM13_S03 = calculateDTW(df_LSM13_S03['Patient'], df_LSM13_S03['Therapist'])

# calculate the Pearson correlation between the two series
#pearson_MP15_S11 = df_MP15_S11['Patient'].corr(df_MP15_S11['Therapist'])
#pearson_MP15_S12 = df_MP15_S12['Patient'].corr(df_MP15_S12['Therapist'])
#pearson_MP15_S13 = df_MP15_S13['Patient'].corr(df_MP15_S13['Therapist'])
#pearson_MP15_S14 = df_MP15_S14['Patient'].corr(df_MP15_S14['Therapist'])
#pearson_MP15_S15 = df_MP15_S15['Patient'].corr(df_MP15_S15['Therapist'])
#pearson_MP15_S16 = df_MP15_S16['Patient'].corr(df_MP15_S16['Therapist'])
#pearson_MP15_S18 = df_MP15_S18['Patient'].corr(df_MP15_S18['Therapist'])
#pearson_MP15_S19 = df_MP15_S19['Patient'].corr(df_MP15_S19['Therapist'])
#pearson_MP15_S20 = df_MP15_S20['Patient'].corr(df_MP15_S20['Therapist'])
#pearson_MP15_S21 = df_MP15_S21['Patient'].corr(df_MP15_S21['Therapist'])
pearson_MP15_S22 = df_MP15_S22['Patient'].corr(df_MP15_S22['Therapist'])
pearson_HHC07_S06 = df_HHC07_S06['Patient'].corr(df_HHC07_S06['Therapist'])
pearson_LSM13_S03 = df_LSM13_S03['Patient'].corr(df_LSM13_S03['Therapist'])

# make a dataframe with the results
data = {'Dyad': ['MP15_S22, HHC07_S06, LSM13_S03'],
        'DTW Distance': [dtw_MP15_S22.distance, dtw_HHC07_S06.distance, dtw_LSM13_S03.distance],
        'Pearson Correlation': [pearson_MP15_S22, pearson_HHC07_S06, pearson_LSM13_S03],
        'DTW Normalized Distamce': [dtw_MP15_S22.normalizedDistance, dtw_HHC07_S06.normalizedDistance, dtw_LSM13_S03.normalizedDistance]}

# Assuming all these lists have the same length
dyads = ['MP15_S22', 'HHC07_S06', 'LSM13_S03']
dtw_distances = [dtw_MP15_S22.distance, dtw_HHC07_S06.distance, dtw_LSM13_S03.distance]
pearson_correlations = [pearson_MP15_S22, pearson_HHC07_S06, pearson_LSM13_S03]
dtw_normalized_distances = [dtw_MP15_S22.normalizedDistance, dtw_HHC07_S06.normalizedDistance, dtw_LSM13_S03.normalizedDistance]

# Check if all lists have the same length
assert len(dyads) == len(dtw_distances) == len(pearson_correlations) == len(dtw_normalized_distances), "Lists have different lengths!"

# Create the dictionary with consistent lengths
data = {
    'Dyad': dyads,
    'DTW Distance': dtw_distances,
    'Pearson Correlation': pearson_correlations,
    'DTW Normalized Distance': dtw_normalized_distances
}
print(data)
df_results = pd.DataFrame(data)
# save the results to a csv file
df_results.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/DTW3_results.csv', index=False)
