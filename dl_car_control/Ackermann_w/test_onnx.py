#!/usr/bin/env python
import os
import numpy as np
import cv2
import onnx
import onnxruntime as ort
import time
import utils.hal as HAL
import matplotlib.pyplot as plt

ort_session = ort.InferenceSession("mynet.onnx",providers=['CPUExecutionProvider'])

def user_main():

    image= HAL.getImage()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # height, width
    height = image.shape[0]
    width = image.shape[1]
    input_size =[66, 200]
    #crop image
    if height > 100:
        cropped_image = image[240:480, 0:640]
        resized_image = cv2.resize(cropped_image, (input_size[1], input_size[0]))
        img = cv2.resize(img, (int(200), int(66)))
        # Display cropped image

        #cv2.imshow("cropped", resized_image)
        #cv2.waitKey(1)
        input_tensor = resized_image.reshape((1, 3, input_size[0], input_size[1])).astype(np.float32)
        # Inference (min 20hz max 200hz)
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        output = ort_session.run(None, ort_inputs)
        
        HAL.setV(output[0])
        HAL.setW(output[1])



def main():

    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)

# Execute!
if __name__ == "__main__":
    main()
