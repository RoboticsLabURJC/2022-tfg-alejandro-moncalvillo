#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import time
import cv2
import torch
from torchvision import transforms
import utils.hal as HAL
from utils.pilotnet import PilotNet
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", type=str, default= None, help="Directory to find Test Data")
    parser.add_argument("--value", type=int, default=1, help="")


    args = parser.parse_args()
    return args

def main():


    args = parse_args()
    path = os.getcwd()
   
    data_file = open(path + "/" + args.test_dir + "/data.csv", "r")
    reader_csv = csv.reader(data_file) 

    writer_output = csv.writer(open(path + "/" + args.test_dir  + "/new_data.csv", "w"))

    writer_output.writerow(["image_name","v","w"])

    first_line = True
    total_time = 0
    min = 20000
    max = -1

    total_loss_v = 0
    total_loss_w = 0
    for line in reader_csv:
        
        if first_line:
            first_line = False
            continue
        writer_output.writerow([line[0],line[1],float(line[2])*args.value])

        
    
    data_file.close()


    print("FIN")



# Execute!
if __name__ == "__main__":
    main()
