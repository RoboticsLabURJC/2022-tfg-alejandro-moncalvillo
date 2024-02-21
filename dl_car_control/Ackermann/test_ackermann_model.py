#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse
import onnx
import onnxruntime as ort
import time
import utils.hal as HAL
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from utils.pilotnet import PilotNet


#ort_session = ort.InferenceSession("mynet_holo.onnx",providers=['CPUExecutionProvider'])

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--net_name", type=str, default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")


    args = parser.parse_args()
    return args

args = parse_args()

def user_main():

    image= HAL.getImage()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # height, width
    height = image.shape[0]
    width = image.shape[1]
    device = torch.device("cpu")
    image_shape = (66, 200, 3)
    num_labels = 2
    input_size =[66, 200]

    pilotModel = PilotNet(image_shape, num_labels).to(device)
    pilotModel.load_state_dict(torch.load(args.net_name,map_location=device))
    pilotModel.eval()
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ]) 
    #crop image
    if height > 100:
        
        cropped_image = image[240:480, 0:640]

        resized_image = cv2.resize(cropped_image, (int(input_size[1]), int(input_size[0])))

        """
        # Display cropped image

        #cv2.imshow("cropped", resized_image)
        #cv2.waitKey(1)
        input_tensor = resized_image.reshape((1, 3, input_size[0], input_size[1])).astype(np.float32)
        # Inference (min 20hz max 200hz)
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        output = ort_session.run(None, ort_inputs)[0][0]
        #print(output)
        """



        input_tensor = preprocess(resized_image).to(device)
        input_batch = input_tensor.unsqueeze(0)
        output = pilotModel(input_batch)
        v = output[0].detach().numpy()[0]
        w = output[0].detach().numpy()[1]
        print(v)
        print(w)
        
        HAL.setV(v)
        HAL.setW(w)




def main():

    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)

# Execute!
if __name__ == "__main__":
    main()
