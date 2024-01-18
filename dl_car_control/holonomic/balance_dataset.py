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
    parser.add_argument('-f','--directory', nargs='+', help='Name of the directory with the data to balance', required=True)

    args = parser.parse_args()
    return args

def random_num_list(num_range, size):

    return random.sample(range(num_range), size)

#Returns a list where the first element is the index of the largest number
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

#Balances the dataset contained in the directory passed, it must have a data.csv file
def balance_single_dataset(directory):
    path = os.getcwd()
    old_file = path + "/" + directory + "/old_data.csv"
    new_file = path + "/" + directory + "/data.csv"

    if not os.path.exists(new_file):
        print("data.csv file not found in " + directory)
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
    
    print("Balanced data.csv file in directory " + directory)
    return True

#Removes the images not listed in the data.csv file
def remove_old_images(directory):

    path = os.getcwd()
    data_path = path + "/" + directory + "/data.csv"
    names_list = []
    with open(data_path) as data_file:
        header = next(data_file)
        reader = csv.reader(data_file)
        for row in reader: 
            names_list.append(row[0]) 

    image_ext = ".png"

    # Iterate over files in the directory
    for file_name in os.listdir(path + "/" + directory):

        if image_ext in file_name  and file_name not in names_list:

            file_path = path + "/" + directory + "/" + file_name
            os.remove(file_path)

    print("Removed old images in directory " + directory)

#For each image in the data.csv file creates a mirrored one and
#appends
def generate_mirrored_images(directory):

    path = os.getcwd()
    data_path = path + "/" + directory + "/data.csv"
    mirrored_rows = []

    with open(data_path) as data_file:
        header = next(data_file)
        reader = csv.reader(data_file)
        for row in reader:
            original_image = cv2.imread(path + "/" + directory + "/" +row[0])
            mirrored_image = cv2.flip(original_image, 1)
            image_name =row[0].split('.')[0]
            cv2.imwrite(path + "/" + directory + "/" + "-" + image_name + ".png", mirrored_image)
            mirrored_rows.append(["-" + image_name + ".png" ,row[1],str(-float(row[2]))])

    with open(data_path, 'a', newline='') as data_file:  
        writer = csv.writer(data_file)
        for element in mirrored_rows:
            writer.writerow(element)
    print("Generated mirrored images in directory " + directory)

def main():
    args = parse_args()
    for directory in args.directory:
        if not balance_single_dataset(directory):
            pass
        remove_old_images(directory)
        generate_mirrored_images(directory)




# Execute!
if __name__ == "__main__":
    main()