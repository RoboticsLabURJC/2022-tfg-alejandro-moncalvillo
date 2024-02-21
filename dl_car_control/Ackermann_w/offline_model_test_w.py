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
    parser.add_argument("--net_name", type=str, default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")


    args = parser.parse_args()
    return args

def main():

    # Device Selection (CPU/GPU)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"


    data_w_array= []
    net_w_array= []
    n_array = []
    count = 1

    args = parse_args()
    path = os.getcwd()
    image_shape = (66, 200, 3)
    num_labels = 1
    input_size =[66, 200]

    pilotModel = PilotNet(image_shape, num_labels).to(device)
    pilotModel.load_state_dict(torch.load(args.net_name,map_location=device))
    pilotModel.eval()

    preprocess = transforms.Compose([
        # convert the frame to a CHW torch tensor for training
        transforms.ToTensor()
    ]) 

    data_file = open(path + "/" + args.test_dir + "/data.csv", "r")
    reader_csv = csv.reader(data_file) 

    first_line = True
    total_time = 0
    min = 20000
    max = -1
    total_loss_w = 0
    for line in reader_csv:
        
        if first_line:
            first_line = False
            continue
                    
            
        image = cv2.imread(path + "/" + args.test_dir + "/" + line[0])

        start_time = datetime.now()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # height, width
        height = image.shape[0]
        width = image.shape[1]
        
        #crop image
        cropped_image = image[240:480, 0:640]

        resized_image = cv2.resize(cropped_image, (int(input_size[1]), int(input_size[0])))

        # Display cropped image
        #cv2.imshow("image", resized_image)       
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        input_tensor = preprocess(resized_image).to(device)

        #print(type(input_tensor))
        # The model can handle multiple images simultaneously so we need to add an
        # empty dimension for the batch.
        # [3, 200, 66] -> [1, 3, 200, 66]
        input_batch = input_tensor.unsqueeze(0)
        # Inference (min 20hz max 200hz)

        output = pilotModel(input_batch)
        
        #print(output)
        if device_type == "cpu":
        
            net_w_array.append((output[0].detach().numpy()[0]))

            total_loss_w = total_loss_w + abs(float(line[1])-output[0].detach().numpy()[0]) 

        else:
        
            net_w_array.append(output.data.cpu().numpy()[0][0])

   
            total_loss_w = total_loss_w +  abs(float(line[1])-output.data.cpu().numpy()[0][0]) 

        data_w_array.append(float(line[1]))
        n_array.append(count)
        
        finish_time = datetime.now()
        dt = finish_time - start_time
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        total_time = total_time + ms
        if ms < min:
            min = ms
        if ms > max:
            max = ms

        count = count + 1
        

    
    data_file.close()

    print("Tiempo medio:"+str(total_time/count))
    print("Error medio w:"+str(total_loss_w/count))
    print("Tiempo min:"+str(min))
    print("Tiempo max:"+str(max))
    plt.plot(n_array, data_w_array, label = "controller", color='b')
    plt.plot(n_array, net_w_array, label = "net", color='tab:orange')

    plt.show()

    print("FIN")



# Execute!
if __name__ == "__main__":
    main()
