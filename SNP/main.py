import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from segmentar import Get_Images
from tensorflow.keras.models import load_model

model_emoji = load_model('emoji_model.h5')
model = load_model('only_alphabet_model.h5')
def Get_Ans(image):
    class_mapping='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    class_mapping1='1234567'
    images, rect=Get_Images(image)
    answer=""
    for i in range(len(rect)):
        if(len(rect[i])==4):
          image1=images[rect[i][2]:rect[i][3], rect[i][0]:rect[i][1]]
        else:
          continue
        image1=cv2.copyMakeBorder(image1,40,40,40,40,cv2.BORDER_CONSTANT)
        image1 = cv2.resize(image1, (28, 28))
        image1 = (np.array(image1)).reshape(1 , 28 , 28 , 1)
        image1=image1/255

        result = np.argmax(model.predict(image1))
        result_confidence=np.max(model.predict(image1))

        result1 = np.argmax(model_emoji.predict(image1))
        result_confidence1=np.max(model_emoji.predict(image1))
        if(result_confidence1>result_confidence and result_confidence<0.99):
          answer+=(class_mapping1[result1])
        else :
          answer+=(class_mapping[result])
    return answer
if __name__=="__main__":
    print("Input the path of your image")

    while True:
        path = input()
        if not os.path.exists(path):
            print("please enter Correct Path")
            continue
        img = cv2.imread(path)
        ans = Get_Ans(img)
        print(ans)
        print("Enter another path to test")
  
