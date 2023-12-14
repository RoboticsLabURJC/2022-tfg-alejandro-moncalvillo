#!/usr/bin/env python
import os
import numpy as np
import cv2
import onnx
import onnxruntime as ort
import time
import utils.hal as HAL
import matplotlib.pyplot as plt
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", type=str, default= None, help="Directory to find Test Data")
    parser.add_argument("--net_name", type=str, default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")


    args = parser.parse_args()
    return args

def main():


    data_v_array= []
    data_w_array= []
    net_v_array= []
    net_w_array= []
    n_array = []
    count = 1

    args = parse_args()
    path = os.getcwd()

    data_file = open(path + "/" + args.test_dir + "/data.csv", "r")
    reader_csv = csv.reader(data_file) 
    ort_session = ort.InferenceSession(path + "/" + args.net_name,providers=['CPUExecutionProvider'])
    
    first_line = True

    for line in reader_csv:
        
        if first_line:
            first_line = False
            continue
        
        image = cv2.imread(path + "/" + args.test_dir + "/" + line[0])
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # height, width
        height = image.shape[0]
        width = image.shape[1]
        input_size =[66, 200]
        #crop image
        cropped_image = image[240:480, 0:640]

        resized_image = cv2.resize(cropped_image, (input_size[1], input_size[0]))

        # Display cropped image
        #cv2.imshow("image", resized_image)       
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        input_tensor = resized_image.reshape((1, 3, input_size[0], input_size[1])).astype(np.float32)
        # Inference (min 20hz max 200hz)
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        output = ort_session.run(None, ort_inputs)[0][0]
        
        net_v_array.append(output[0])
        net_w_array.append(output[1])

        data_v_array.append(float(line[1]))

        data_w_array.append(float(line[2]))

        n_array.append(count)
        
        count = count + 1

    data_file.close()

    plt.subplot(1, 2, 1)
    plt.plot(n_array, data_v_array, label = "controller", color='b')
    plt.plot(n_array, net_v_array, label = "net", color='tab:orange')

    plt.subplot(1, 2, 2)
    plt.plot(n_array, data_w_array, label = "controller", color='b')
    plt.plot(n_array, net_w_array, label = "net", color='tab:orange')

    plt.show()

    print("FIN")



# Execute!
if __name__ == "__main__":
    main()
