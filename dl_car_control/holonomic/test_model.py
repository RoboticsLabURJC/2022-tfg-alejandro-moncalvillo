#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse
import time
import utils.hal_holonomic as HAL
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import utils.hal_holonomic as HAL
from inference_model import GetModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="name of the model")


    args = parser.parse_args()
    return args

args = parse_args()

def user_main():

    image= HAL.getImage()
    height = image.shape[0]
    device = 'cuda'
    input_size = [16,12]
    model = GetModel(model_path=args.model_path,device=device)
    #crop image
    if height > 100:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = np.array(cv2.resize(image, (int(input_size[0]), int(input_size[1]))))/255.0


        v, w = model(im)

        HAL.setV(v)
        HAL.setW(w)




def main():

    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)

# Execute!
if __name__ == "__main__":
    main()
