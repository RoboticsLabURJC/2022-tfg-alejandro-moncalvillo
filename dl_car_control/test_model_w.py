#!/usr/bin/env python
import os
import numpy as np
import time
import argparse
import utils.hal as HAL
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import transforms
import utils.hal as HAL
from utils.pilotnet import PilotNet

image_shape = (66, 200, 3)
num_labels = 1
# Device Selection (CPU/GPU)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
pilotModel = PilotNet(image_shape, num_labels).to(device)

preprocess = transforms.Compose([
    # convert the frame to a CHW torch tensor for training
    transforms.ToTensor()
]) 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--net_name", type=str, default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")


    args = parser.parse_args()
    return args


def user_main():

    image= HAL.getImage()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # height, width
    height = image.shape[0]
    width = image.shape[1]
    input_size =[66, 200]
    #crop image
    if height > 100:

        #crop image
        cropped_image = image[240:480, 0:640]

        resized_image = cv2.resize(cropped_image, (int(input_size[1]), int(input_size[0])))

        input_tensor = preprocess(resized_image).to(device)

        #print(type(input_tensor))
        # The model can handle multiple images simultaneously so we need to add an
        # empty dimension for the batch.
        # [3, 200, 66] -> [1, 3, 200, 66]
        input_batch = input_tensor.unsqueeze(0)
        # Inference (min 20hz max 200hz)

        output = pilotModel(input_batch).detach().numpy()
        
        #print(output[0])
        HAL.setV(4)
        HAL.setW(output[0])



def main():

    HAL.setW(0)
    HAL.setV(0)
    args = parse_args()


    pilotModel.load_state_dict(torch.load(args.net_name,map_location=device))
    pilotModel.eval()

    HAL.main(user_main)

# Execute!
if __name__ == "__main__":
    main()
