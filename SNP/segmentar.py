import cv2
import numpy as np


def Get_Images(img) : 
  kernel = np.ones((5,5))
  gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	                                     cv2.THRESH_BINARY_INV, blockSize = 321, C = 80)
  gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
  gray_img = cv2.dilate(gray_img,kernel,iterations = 3)
  gray_img = cv2.erode(gray_img,kernel,iterations = 1)
  gray_image = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
  lst=[]
  for j in range(gray_image.shape[1]):
    sum = 0
    for i in range(gray_image.shape[0]):
      sum = sum+gray_image[i][j]
    lst.append(sum)
  rect = []
  f = 0
 
  for i in range(len(lst)):
    if lst[i] != 0 and f==0:
      x1=i
      f=1
    elif lst[i] == 0 and f==1:
      x2=i
      if(abs(x1-x2)>30):
        rect.append([min(x1,x2),max(x1,x2)])
      f=0 
  for k in range(len(rect)):
    x1=rect[k][0]
    x2=rect[k][1]
    lst2=[]
    for i in range(gray_image.shape[0]):
      sum=0
      for j in range(min(x1,x2),max(x1,x2)+1):
        sum = sum+gray_image[i][j]
      lst2.append(sum)
    f=0
    for i in range(len(lst2)):
      if lst2[i] != 0 and f==0:
        y1=i
        f=1
      if lst2[i] == 0 and f==1:
        y2=i
        if(abs(y2-y1)>30):
          rect[k].append(min(y1,y2))
          rect[k].append(max(y1,y2))
        f=0 
  
  return gray_img, rect