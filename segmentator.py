import cv2
import numpy as np
import os
import scipy.ndimage





def Get_Images(img) : 
  lstimgs=[]
  lst=[]
  gray_image = img
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
      if(abs(x1-x2)>5):
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
        if(abs(y2-y1)>5):
          rect[k].append(min(y1,y2))
          rect[k].append(max(y1,y2))
          lstimgs.append(img[min(y1,y2):max(y1,y2) , x1:x2])
        f=0 
  
  return lstimgs




def getMeanArea(contours):
    meanArea=0
    for contour in contours:
        meanArea+=cv2.contourArea(contour)
    meanArea=(meanArea)/len(contours)
    return meanArea
    
    
def getRatioArea(contours):
    meanArea=0
    for contour in contours:
        meanArea+=cv2.contourArea(contour)
    cnsSorted = sorted(contours, key=lambda x:cv2.contourArea(x), reverse = True)
    ratioArea = cv2.contourArea(cnsSorted[0])/meanArea
    return ratioArea



def purify(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (5 , 5))
    img = cv2.dilate(img,(5 , 5),iterations = 3)
    img = cv2.erode(img,(5 , 5),iterations = 1)
    imglst = Get_Images(img)
    maxarea=0

    for i in range(len(imglst)):
      im1 = cv2.copyMakeBorder(imglst[i],32,32,32,32,cv2.BORDER_CONSTANT)
      nlabels,labels,stats,centroids=cv2.connectedComponentsWithStats(im1,None,None,None,8,cv2.CV_32S)
      for j in range(1 , nlabels):
        if maxarea<stats[j , cv2.CC_STAT_AREA]:
          maxarea = stats[j , cv2.CC_STAT_AREA]
    
    if maxarea<450:
      return np.array([[1]])
    for i in range(len(imglst)):
      im1 = cv2.copyMakeBorder(imglst[i],32,32,32,32,cv2.BORDER_CONSTANT)
      nlabels,labels,stats,centroids=cv2.connectedComponentsWithStats(im1,None,None,None,8,cv2.CV_32S)
      for j in range(1 , nlabels):
        if maxarea==stats[j , cv2.CC_STAT_AREA]:
          im1 = cv2.dilate(im1,(5 , 5),iterations = 10)
          im1 = cv2.erode(im1,(5 , 5),iterations = 2)
          return cv2.resize(im1 , (28 , 28))



def extract_character(image, recursion = 0):
    all_imgs = []
    thresh = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    newim = thresh
    pad=5
    thresh=cv2.GaussianBlur(thresh, (3,3), 0)
    thresh = cv2.adaptiveThreshold(thresh, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    kernel1 = np.ones((10,10), np.uint8)
    thresh = cv2.dilate(thresh, kernel1, iterations = 1)
    if(recursion<3):
    	thresh2 = cv2.erode(thresh, np.ones((2,2), np.uint8), iterations = 2)
    	thresh2 = scipy.ndimage.median_filter(thresh2, (5, 1)) # remove line noise
    	thresh2 = scipy.ndimage.median_filter(thresh2, (1, 5)) # weaken circle noise
    	thresh2 = scipy.ndimage.median_filter(thresh2, (5, 1)) # remove line noise
    	thresh2 = scipy.ndimage.median_filter(thresh2, (1, 5)) # weaken circle noise
    	contours1, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
    	contours1, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords=[]
    count=0
    ratioArea = getRatioArea(contours1)
    if(ratioArea<0.3 or recursion>2):
    	kernel2 = np.ones((2,2), np.uint8)
    elif(ratioArea>0.85 and recursion<2):
    	kernel2 = np.ones((5,5), np.uint8)
    else:
    	kernel2 = np.ones((3,3), np.uint8)
    if(ratioArea > 0.3 and recursion<2):
    	thresh = cv2.erode(thresh, kernel2, iterations = 2)
    	thresh = scipy.ndimage.median_filter(thresh, (5, 1)) # remove line noise
    	thresh = scipy.ndimage.median_filter(thresh, (1, 5)) # weaken circle noise
    	thresh = scipy.ndimage.median_filter(thresh, (5, 1)) # remove line noise
    	thresh = scipy.ndimage.median_filter(thresh, (1, 5)) # weaken circle noise
    thresh = cv2.dilate(thresh, kernel1, iterations = 1)
    #cv2.imshow('thresh',thresh)
    #cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords=[]
    count=0
    meanArea=getMeanArea(contours)
    for contour in contours:
      
        if cv2.contourArea(contour)<500 :
            continue

        (x,y,w,h)=cv2.boundingRect(contour)
        if cv2.contourArea(contour)>0.05*meanArea:
            if w / h > 1.9:
                #Split it in half into two letter regions
                half_width = int(w / 2)
                coords.append((x, y, half_width, h))
                coords.append((x + half_width, y, half_width, h))
                count=count+2
            else:  
                coords.append((x, y, w, h))
                count=count+1
    coords=sorted(coords,key=lambda x: x[0])
    img_paths=[]
    if(count >7 and recursion <3):
    	img_array = extract_character(image, recursion + 1)
    	return img_array
    else:
      for i in range(count):
        result=(newim[coords[i][1]:coords[i][1]+coords[i][3],coords[i][0]:coords[i][0]+coords[i][2]])
        #cv2.imshow('result',result)
        #cv2.waitKey(0)
        # filename='character'+str(i)+'.jpeg'
        # cv2.imwrite(filename,cv2.bitwise_not(result))
        # result = cv2.threshold(result, 0, 255,
	      #           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        result = cv2.adaptiveThreshold(result, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        result = purify(result)
        if result.shape==(1 , 1):
          continue
        all_imgs.append(np.array(result))
        # img_paths.append(filename)
      return np.array(all_imgs)