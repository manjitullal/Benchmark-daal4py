import matplotlib.pyplot as plt
import glob
import os


#Get dataset
path = '../data/'
files = os.listdir(path)
print("Datasets available \n")
for f in files:
	print(f)

key = input('Which Data do you want to generate plots for? \n')

searchSerial = key + "_serial"
searchParallel = key + "_parallel"

temp_folders = []
for name in glob.glob('../temp/*'):
    if key in name:
        temp_folders.append(name)

for folder in temp_folders:
    
    if classification:
        X_NB = []
    if searchSerial in folder:
        dict_path = folder + '/latency_stats.json'
        with open(dict_path) as f:
            serial_performance = json.load(f)
            
            