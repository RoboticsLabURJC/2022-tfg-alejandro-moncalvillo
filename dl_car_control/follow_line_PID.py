#!/usr/bin/env python
import os

import numpy as np
import cv2
import time
import hal as HAL
import csv


prev_err = 0
prev_prev_err = 0
prev_angular_control = 0
prev_linear_control = 0

max_linear_vel = 7
Ts = 1/30
iteration = 0
writer_output = csv.writer(open(os.getcwd() + "/output_montmelo.csv", "w"))
def user_main():
    global prev_err
    global prev_prev_err
    global prev_angular_control
    global prev_linear_control
    global iteration
    inferior= np.array([0, 0, 100])
    superior= np.array([50,50 , 255])
    image= HAL.getImage()
    # get dimensions of image
    dimensions = image.shape
    
    # height, width

    height = image.shape[0]
    width = image.shape[1]
    image= HAL.getImage()
    mascara=cv2.inRange(image, inferior, superior)
    filtered_image=cv2.bitwise_and(image, image, mask=mascara)
    
    # convert image to grayscale image
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    
    # convert the grayscale image to binary image
    
    ret,thresh = cv2.threshold(gray_image,0,255,0)
    
    # calculate moments of binary image
    M = cv2.moments(thresh)
    
    if M["m00"] != 0:
        #coordinates of the centroid center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        
        #cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
        #cv2.putText(image, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        err= width/2 - cX
        
        #angular vel control
        
        K_p=0.0005
        K_d=0.00001
        K_i=0.00000
        
        angular_control=prev_angular_control+(K_p+K_i*(Ts/2)+K_d/Ts)*err +(-K_p+K_i*(Ts/2)-2*(K_d/Ts))*prev_err+(K_d/Ts)*prev_prev_err
        HAL.setW(angular_control)
        
    
        #linear vel control
        linear_K_p=0.01
        linear_K_d=0.005
    
        
        linear_control=max_linear_vel - abs(linear_K_p*err) - abs(linear_K_d *(err-(prev_err-prev_prev_err)))
        
        HAL.setV(abs(linear_control))
        
    

        #cv2.putText(image, "err: "+str(err), (50,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #cv2.imshow("Imagen", image)
        prev_prev_err=prev_err
        prev_err=err
        prev_angular_control=angular_control
        #crop image
        cropped_image = image[((height//2)+1):height, 0:width]
        resized_image = cv2.resize(cropped_image, (200, 60))
        # Display cropped image
        cv2.imshow("cropped", resized_image)
        cv2.waitKey(1)
        
        cv2.imwrite(os.getcwd() +'/dataset_montmelo/image' + str(iteration) + '.jpg', resized_image)
        

        iteration = iteration + 1

        #write to file
        writer_output.writerow(['image' + str(iteration) + '.jpg',abs(linear_control),angular_control])

    else:
        HAL.setW(0)
        #HAL.setV(1)

def main():

    #HAL.setW(0)
    #HAL.setV(max_linear_vel)
    HAL.main(user_main)

# Execute!
if __name__ == "__main__":
    main()
