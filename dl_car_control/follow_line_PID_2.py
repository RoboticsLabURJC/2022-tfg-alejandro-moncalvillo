#!/usr/bin/env python
import os

import numpy as np
import cv2
import time
import hal as HAL
import csv

redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)
redBajo2=np.array([175, 100, 20], np.uint8)
redAlto2=np.array([179, 255, 255], np.uint8)
Kp_recta = 0.001
Kp_curva = 0.0005
Kd_curva = 0.005
Kd_recta = 0.0028
V_recta = 6
V_curva = 5
last_error = 0
iteration = 0

writer_output = csv.writer(open(os.getcwd() + "/output.csv", "w"))

def user_main():
    global redBajo1 
    global redAlto1
    global redBajo2
    global redAlto2
    global Kp_recta
    global Kp_curva
    global Kd_curva
    global Kd_recta
    global last_error
    global iteration
    global V_recta
    global V_curva
    
    img = HAL.getImage()
    height, width = img.shape[:2]
    
    frameHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
    maskRed = cv2.add(maskRed1, maskRed2)
    maskRedvis = cv2.bitwise_and(img, img, mask= maskRed)
    filtered = cv2.bitwise_and(gray, gray, mask = maskRed)
    
    
    ret,thresh = cv2.threshold(filtered,20,200,0)
    
    # calculate moments of binary image
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # put text and highlight the center
        #cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)       
        error = (width/2 - cX)
        linear_control=V_recta-abs(error)*Kp_recta
        angular_control=Kp_recta*error + Kd_recta*(error - last_error)
        #if abs(error) < 180:
            #linear_control=V_recta-abs(error)*Kp_recta
            #angular_control=Kp_recta*error + Kd_recta*(error - last_error)
        #else:
            #linear_control=V_curva-abs(error)*Kp_curva
            #angular_control=Kp_curva*error + Kd_curva*(error - last_error)

        HAL.setV(linear_control)
        HAL.setW(angular_control)
        last_error = error
  
        
        # display the image
        #crop image
        cropped_image = img[((height//2)+1):height, 0:width]
        resized_image = cv2.resize(cropped_image, (200, 60))
        # Display cropped image
        cv2.imshow("cropped", resized_image)
        cv2.waitKey(1)
        """
        cv2.imwrite(os.getcwd() +'/dataset/image' + str(iteration) + '.jpg', resized_image)

        iteration = iteration + 1

        #write to file
        writer_output.writerow(['image' + str(iteration) + '.jpg',abs(linear_control),angular_control])
        """
        

def main():

    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)

# Execute!
if __name__ == "__main__":
    main()
