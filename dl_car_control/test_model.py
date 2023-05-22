#!/usr/bin/env python
import os
import numpy as np
import cv2
import onnx
import onnxruntime as ort
import time
import hal as HAL
import matplotlib.pyplot as plt

ort_session = ort.InferenceSession("first_mynet.onnx",providers=['CPUExecutionProvider'])

def user_main():

    image= HAL.getImage()

    # height, width
    height = image.shape[0]
    width = image.shape[1]
    input_size =[239, 640]
    #crop image
    if height > 100:
        cropped_image = image[((height//2)+1):height, 0:width]
        # Display cropped image

        cv2.imshow("cropped", cropped_image)
        cv2.waitKey(1)
        input_tensor = cropped_image.reshape((1, 3, input_size[0], input_size[1])).astype(np.float32)
        # Inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        output = ort_session.run(None, ort_inputs)[0][0]
        print(output)

        V_pred = output[0]
        W_pred = output[1]

        HAL.setV(V_pred)
        HAL.setW(W_pred)




def main():

    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)

# Execute!
if __name__ == "__main__":
    main()
