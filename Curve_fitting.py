import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def rgb_to_gray(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
def rgb_to_hls(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
def rgb_to_hsv(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

def plt_show(image):
    plt.imshow(image)
    plt.show()

img = cv2.imread('circle1.jpg')
img = cv2.resize(img,(800,400))
output = img.copy()

plt.imshow(img)
plt.show()

img =bgr_to_rgb(img)
gray = rgb_to_gray(img)

plt.imshow(gray)
plt.show()

def perspective_transform(image):
    src = np.float32([[250,400],[250,270],[550,270],[550,400]])
    dst = np.float32([[250,400],[250,0],[550,0],[550,400]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size=(image.shape[1],image.shape[0])
    print(M.shape,Minv.shape)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped,Minv,M

binary_warped,Minv,M = perspective_transform(img)
plt.imshow(binary_warped)
plt.show()

warped_copy = binary_warped.copy()
binary_warped_hsv = rgb_to_hsv(binary_warped)

plt.imshow(binary_warped)
plt.show()

def show_image(image):
    cv2.imshow("picture",image)
    cv2.waitKey(1000)
binary_warped_hsv_blur = cv2.GaussianBlur(binary_warped_hsv,(3,3),0)
hsv_blur_h = binary_warped_hsv_blur[:,:,0]
ret,binary_threshold = cv2.threshold(hsv_blur_h,80,90,cv2.THRESH_BINARY)
show_image(binary_threshold)
plt.imshow(binary_threshold)
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
dilate = cv2.dilate(binary_threshold,kernel,iterations = 2)
plt.imshow(dilate)
plt.show()
erode = cv2.erode(binary_threshold,kernel,iterations = 1)
plt_show(erode)

	
