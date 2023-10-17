import os
import numpy as np
import time
import cv2
import csv
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default=None, help="To save or not to save the data")
    parser.add_argument('-f','--folder', nargs='+', help='Name of the folder with the data to balance', required=True)

    args = parser.parse_args()
    return args

def random_num_list(num_range, size):

    return random.sample(range(num_range), size)

#returns a list where the first element is the index of the largest number
#and the second element is the index of the second largest number
def max_index_two(data):
    mx = max(data[0], data[1]) 
    secondmax = min(data[0], data[1])
    n = len(data)
    for i in range(2,n): 
        if data[i] > mx: 
            secondmax = mx
            mx = data[i] 
        elif data[i] > secondmax and mx != data[i]: 
            secondmax = data[i]
        elif mx == secondmax and secondmax != data[i]:
            secondmax = data[i]

    index_list = [-1,-1]

    for j in range(n):
        if data[j] == mx:
            index_list[0] = j
        if data[j] == secondmax:
            index_list[1] = j
    
    return index_list
            

def balance_single_dataset(folder):
    path = os.getcwd()
    old_file = path + "/" + folder + "/old_data.csv"
    new_file = path + "/" + folder + "/data.csv"

    if not os.path.exists(new_file):
        print("data.csv file not found in " + folder)
        return False

    os.rename(new_file, old_file)

    old_csv = pd.read_csv(old_file)

    types = old_csv.groupby('v').size().axes[0]
    types_sizes = old_csv.groupby('v').size().values

    sizes_index = max_index_two(types_sizes)
    max_index = sizes_index[0]
    second_max_index = sizes_index[1]

    max_rows = old_csv.loc[old_csv['v'] == types[max_index]]

    other_rows = old_csv.loc[old_csv['v'] != types[max_index]]
    other_rows.to_csv(new_file,index=False)

    rand_list = random_num_list(types_sizes[max_index],types_sizes[second_max_index])
    
    writer_output = csv.writer(open(new_file, "a"))

    for i in range(types_sizes[second_max_index]):
        rand_index = rand_list[i]
        image_name = max_rows.iloc[[rand_index]]['image_name'].values[0]
        vel = max_rows.iloc[[rand_index]]['v'].values[0]
        ang_vel = max_rows.iloc[[rand_index]]['w'].values[0]
        writer_output.writerow([image_name,vel,ang_vel])

    print("Balanced data.csv file in folder " + folder)
    return True



def main():
    args = parse_args()
    for folder in args.folder:
        balance_single_dataset(folder)



# Execute!
if __name__ == "__main__":
    main()