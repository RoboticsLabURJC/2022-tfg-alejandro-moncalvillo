import hal as HAL
import cv2
import numpy as np
import time
import math

# PID class
class PID:
    def __init__(self, kp, kd, ki, st):
        
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.last_error = 0
        self.last_output = 0
        self.last_time = 0
        self.st = st
        self.last_sample = time.time()
        self.iterm = 0
    
    def set_lims(self, outmin, outmax):
        self.outmax = outmax
        self.outmin = outmin
        
    def calc(self, error):
            
        out = self.last_output
        
        # Calculate the time which has passed
        diff_time = time.time() - self.last_sample
        
        if (diff_time >= self.st):
            # Derivative part
            diff_error = error - self.last_error
            
            # Integrative part (never higher than max)
            self.iterm += error * self.ki
            if(self.iterm > self.outmax): self.iterm = self.outmax
            elif (self.iterm < self.outmin): self.iterm = self.outmin
                
            # Output (never higher than max)
            out = self.kp * error + self.kd * diff_error + self.iterm
            if(out > self.outmax): out = self.outmax
            elif (out < self.outmin): out = self.outmin
            
                
            # Store info needed for next time
            self.last_error = error
            self.last_output = out
            self.last_sample = time.time()
        
        return out

# Apply the color mask to the raw img and return the filtered image
def filter_img(raw_image, c_mask):
    
    lower = np.array(c_mask[0], dtype = "uint8")
    upper = np.array(c_mask[1], dtype = "uint8")
    
    mask = cv2.inRange(raw_image, lower, upper)
    f_img = cv2.bitwise_and(raw_image, raw_image, mask = mask)
    
    return f_img

# Gets a reference position between (center+offset, center+offset+margin)
def get_line_ref(img, offset, margin):
    
    height = img.shape[0]
    width = img.shape[1]

    center_row = int(height/2) + offset

    c_x = 0
    c_y = 0
    npixels = 0

    for x in range(width):
        for y in range(center_row, center_row + margin):
            # Get pixel val and compare it with values for black
            pixel_val = img[y][x]
            comparison = (pixel_val == black_pixel)
            
            if not comparison.all():
                c_x += x
                c_y += y
                npixels += 1
    
    if (npixels > 0):
        c_x /= npixels
        c_y /= npixels

    return (int(c_x), int(c_y))

# Shows a debug image with the reference point and the set point
def show_debug_img(img, ref):
    
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    
    set_point = (int(width/2), int(height/2))
    
    img = cv2.circle(img, ref, 4, (255, 255, 0), -1)
    # img = cv2.circle(img, set_point, 5, (0, 255, 255), -1)
    img = cv2.line(img, (set_point[0], 0), (set_point[0], height), (0, 255, 0), thickness=1)
    cv2.imshow("imagen", img)
    cv2.waitKey(1)
    #GUI.showImage(img)

# Color filter
red_mask = ([17, 15, 70], [50, 56, 255])
center_offset = 20
center_margin = 10
black_pixel = np.array([0,0,0])
# PID variables
direct = 0

# Angular pid
sp1 = 320
kp_1 = 0.005
kd_1 = 0.0001
ki_1 = 0.0
outmax_1 = 2.5
outmin_1 = -2.5

# Linear pid
kp_2 = 0.03
kd_2 = 0.05
ki_2 = 0.0001
outmax_2 = 4
outmin_2 = -4


# Turn Angular pid
kp_3 = 0.005
kd_3 = 0.0001
ki_3 = 0.0
outmax_3 = 3
outmin_3 = -3

# Turn Linear pid
kp_4 = 0.5
kd_4 = 0.1
ki_4 = 0.01
outmax_4 = 7
outmin_4 = -7

# Car variables
max_linear = 12

# PIDS objects (angular and linear speed)
pid1 = PID(kp_1, kd_1, ki_1, 0.03)
pid1.set_lims(outmin_1, outmax_1)

pid2 = PID(kp_2, kd_2, ki_2, 0.03)
pid2.set_lims(outmin_2, outmax_2)

pid3 = PID(kp_3, kd_3, ki_3, 0.03)
pid3.set_lims(outmin_3, outmax_3)

pid4 = PID(kp_4, kd_4, ki_4, 0.03)
pid4.set_lims(outmin_4, outmax_4)


def user_main():

    raw_img = HAL.getImage()
    f_img = filter_img(raw_img, red_mask)
    raw_img = HAL.getImage()

    height = raw_img.shape[0]

    if height > 100:
        # The reference used for angular speed calculation
        ref1 = get_line_ref(f_img, center_offset, center_margin)
 
        if (ref1 != (0,0)):
            
            # Error calculation
            ref1_x = ref1[0]
            error = sp1 - ref1_x
            
            if abs(error) > 20:
                angular_speed = pid3.calc(error)
                linear_speed = max_linear - abs(pid4.calc(error))
            else:
                angular_speed = pid1.calc(error)
                linear_speed = max_linear - abs(pid2.calc(error))
                
            # Control action
            print (angular_speed)
            HAL.setW(angular_speed)
            HAL.setV(linear_speed)
        else:
            HAL.setW(0)
            HAL.setV(0)


        show_debug_img(f_img, ref1)


def main():
    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)

# Execute!
if __name__ == "__main__":
    main()
