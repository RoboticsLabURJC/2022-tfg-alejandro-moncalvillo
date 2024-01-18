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

    parser.add_argument('-f','--directory', nargs='+', help='Name of the directory with the data to balance', required=True)
    parser.add_argument("--data_ratios", action='append', help="Ratios for each type of data")

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







def create_balanced_dataset(new_directory,old_directory,data,ratios,total_data):
    image_num = 1
    for data_type in data
        for entry in data_type
            image_name = entry[0]
            ang_vel = entry[1]
            writer_output.writerow([image_num,ang_vel])
            image = cv2.imread(directory + image_name)
            cv2.imwrite(str(image_num) + ".png", image)
            image_num= image_num + 1
            
    return
#Balances the dataset contained in the directory passed, it must have a data.csv file
def balance_single_dataset(directory,ratios):
    path = str(os.getcwd())
    
    old_data_file = open(path + "/" + directory + "/data.csv")

    if not os.path.exists(path + "/" + directory + "/data.csv"):
        print("data.csv file not found in " + directory)
        return False
    if not os.path.exists(path + "/" + directory + "_balanced"): 
        # if the circuit folder is not present  
        # then create it. 
        os.makedirs(path + "/" + directory + "_balanced")
    
    balanced_directory = path + "/" + directory + "_balanced"

    csvreader = csv.reader(old_data_file)
    
    header = []
    header = next(csvreader)

    type_1 = []
    type_2 = []
    type_3 = []
    total=0

    for row in csvreader:
        data_w = float(row[1])
        if abs(data_w)< 0.20:
            type_1.append(row)
        elif abs(data_w)> 0.20 and abs(data_w)< 1.00:
            type_2.append(row)
        elif abs(data_w)> 1.00:
            type_3.append(row)
        total = total + 1

    """   
    print("Tipo 1: " + str(len(type_1)/total))
    print("Tipo 2: " + str(len(type_2)/total))
    print("Tipo 3: " + str(len(type_3)/total))
    """ 

    old_data_file.close()

    create_balanced_dataset(path + "/" + directory, balanced_directory, [type_1,type_2,type_3],ratios,total)


    print("Balanced data.csv file in directory: " + balanced_directory)
    
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
def generate_extreme_cases_images(directory):

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
        if not balance_single_dataset(directory,args.data_ratios):
            pass
        




# Execute!
if __name__ == "__main__":
    main()