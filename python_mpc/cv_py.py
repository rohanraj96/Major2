import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import time as time
from math import *

class ComputerVision:
    
    centroids = []
    new_image = np.empty(1)
    
    # Init functions
    def __init__(self):
        centroids = []
        new_image = np.empty(1)
        
    # Helper functions for color space change
    def bgr_to_rgb(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    def rgb_to_gray(self,image):
        return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    def rgb_to_hls(self,image):
        return cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    def rgb_to_hsv(self,image):
        return cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    
    # Helper function to perform perspective transform
    def perspective_transform(self,image):
        src = np.float32([[250,400],[250,270],[550,270],[550,400]])
        dst = np.float32([[250,400],[250,0],[550,0],[550,400]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        img_size=(image.shape[1],image.shape[0])
        warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
        return warped,Minv,M
    
    def show_image(self,image):
        cv2.imshow("picture",image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    
    def findCentroids(self,new_dilate,rgb_warped):
        (_,cnts,_)= cv2.findContours(new_dilate.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print("length of contours",len(cnts))
        for i in range(len(cnts)):
            (x,y),radius = cv2.minEnclosingCircle(cnts[i])
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(rgb_warped,center,radius,(0,255,0),2)
        self.new_image = rgb_warped
        for i in range(len(cnts)):
            M=cv2.moments(cnts[i])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            l=[cx,cy]
            self.centroids.append(l)
            
    def find_slope(self,centers):
        x=[]
        y=[]
        for i in range(len(centers)):
            x.append(centers[i][0])
            y.append(centers[i][1])
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.polyfit(x,y,2)
        return atan(2*z[0]*centers[0][0]+z[1]),z
    
    def tr_2_car(self,centers,car,slope_car):
        for i in range(len(centers)):
            dx=centers[i][0]-car[0]
            dy=centers[i][1]-car[1]

            centers[i][0] = dx
            centers[i][1] = dy
            
            # centers[i][0] = dx*cos(slope_car) + dy*sin(slope_car)
            # centers[i][1] = -dx*sin(slope_car) + dy*cos(slope_car)
            
        return centers

    def tr_pixels_to_cms(self,centers):
        for i in range(len(centers)):
            centers[i][0] *=.026
            centers[i][1] *=.026
        return centers
            
        
    def run(self,img_name):
        # Read the image and perform necessary changes to the image
        # time1 = time.time()

        image = cv2.imread(img_name)
        image= self.bgr_to_rgb(image)
        image = cv2.resize(image,(800,400))
        # self.show_image(image)
        rgb_image=image
        gray_image = self.rgb_to_gray(rgb_image)
        # Perform perspective transform on the image
        rgb_warped,Minv,M = self.perspective_transform(rgb_image)
        gray_warped,Minv,M = self.perspective_transform(gray_image)
        # plt.imshow(rgb_warped)
        rgb_warped_hsv = self.rgb_to_hsv(rgb_warped)
#         plt.imshow(rgb_warped_hsv)
        rgb_warped_hsv_blur = cv2.GaussianBlur(rgb_warped_hsv,(3,3),0)
#         plt.imshow(rgb_warped_hsv_blur)
        rgb_h = rgb_warped_hsv_blur[:,:,0]
#         plt.imshow(rgb_h)
        rgb_s = rgb_warped_hsv_blur[:,:,1]
#         plt.imshow(rgb_s)
        ret,binary_threshold_s = cv2.threshold(rgb_s,80,90,cv2.THRESH_BINARY)
        ret1,binary_threshold_h = cv2.threshold(rgb_h,100,255,cv2.THRESH_BINARY)
        new = cv2.bitwise_and(binary_threshold_s,binary_threshold_h)
#         plt.imshow(new)
        kernel = np.ones((5,5),np.uint8)
        new_dilate = cv2.dilate(new,kernel,iterations=2)
        self.findCentroids(new_dilate,rgb_warped)

        # print(self.centroids)
        
        for i in range(len(self.centroids)):
            self.centroids[i][1]= 400-self.centroids[i][1]

        self.centroids = self.tr_pixels_to_cms(self.centroids)

        # print("centroids in cms",self.centroids)
            
        slope,coeffs = self.find_slope(self.centroids)
        # print("slope",slope)
        
        car=[]
        
        if slope < 0:
            car.append(self.centroids[0][0]+10*.026)
            car.append(self.centroids[0][1]-10*.026)
        else:
            car.append(self.centroids[0][0]-10*.026)
            car.append(self.centroids[0][1]-10*.026)
            
        slope_car = atan(2*coeffs[0]*car[0] + coeffs[1])
        # slope_car1 = atan(2*coeffs[0]*self.centroids[0][0] + coeffs[1]) + 3.14/2
        # print("slope_car",slope_car)
        # print("slope_car1",slope_car1)
        # print("car",car)
        
        centers = self.tr_2_car(self.centroids,car,slope_car)
        # print(centers)

        # print(len(centers))
        # centers = self.tr_pixels_to_cms(centers)
        # print(centers)
        slope,coeffs = self.find_slope(centers)
        print(centers)
        print(coeffs)
        print(slope)

        # time2 = time.time()
        # print(time2-time1)
        
        return coeffs,len(centers)